import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import draw_target_bbox_aabb, draw_target_bbox_obb

class OrientationLoss:
    def __init__(self, init_data, mask_index_info, mllm_parser_orient, args):
        self.device = args.device
        # camera intrinsics info
        cam_info = init_data[0]
        self.cx = cam_info["width"] / 2.0
        self.cy = cam_info["height"] / 2.0
        self.fx = cam_info["fx"]
        self.fy = cam_info["fy"]
        # 3D coordinate info
        self.mask_index = mask_index_info
        
        if mllm_parser_orient is not None:
            # from MLLM
            self.target_ori_delta = []
            for item in mllm_parser_orient:
                azimuth, polar, rotation = item["orient"]
                self.target_ori_delta.append({
                    "azimuth": float(azimuth),
                    "polar": -float(polar),
                    "rotation": -float(rotation)
                })

            target_ori = self.target_pseudo_orientation(init_data, self.target_ori_delta)
            obj_ori = -target_ori # (N,3)
            obj_ori_m = F.normalize(obj_ori.mean(dim=0), dim=0) # (3,)
            self.pseudo_orientation = obj_ori_m

            # camera roll
            self.cam_roll = any(abs(d["rotation"]) > 1e-5 for d in self.target_ori_delta)

    # project to world coordinate
    def project_orientation(self, azimuth, polar, R_c2w, front_dir=None):
        if front_dir is None:
            front_dir = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        # to tensor
        azi = torch.tensor(azimuth, dtype=torch.float32, device=self.device) 
        polar = torch.tensor(polar, dtype=torch.float32, device=self.device) 
        # constraint
        azi = azi % 360.0
        polar = torch.clamp(polar, -90.0, 90.0)

        azi_rad = torch.deg2rad(azi)
        polar_rad = torch.deg2rad(polar)

        x = -1.0 * torch.cos(azi_rad)
        y = -1.0 * torch.tan(polar_rad)
        z = torch.sin(azi_rad)
        camera_location = torch.stack([x, y, z], dim=0)  

        f = camera_location / (camera_location.norm() + 1e-8)
        u_ref = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=self.device)
        if torch.abs(torch.dot(f, u_ref)).item() > 0.99:
            sign = 1.0 if f[1].item() < 0 else -1.0
            u_ref = torch.tensor([sign, 0.0, 0.0], dtype=torch.float32, device=self.device)

        l = torch.cross(u_ref, f, dim=0)
        l = l / (l.norm() + 1e-8)
        u = torch.cross(f, l, dim=0)
        u = u / (u.norm() + 1e-8)

        # view matrix 3x3
        R_o2c = torch.stack([l, u, f], dim=0)
        R_o2w = R_c2w @ R_o2c
        # front and up
        front_w = R_o2w @ front_dir 
        return front_w

    # get front orientation
    def target_pseudo_orientation(self, init_data, target_ori_delta):
        each_score = torch.tensor([cam["score"] for cam in init_data], dtype=torch.float32, device=self.device)
        max_num_objs = max(len(cam["orientation"]) for cam in init_data)
        
        weight_orient = []
        for obj_idx in range(max_num_objs):
            front_list = []
            score_list = []
            for cam_idx, cam in enumerate(init_data):
                if len(cam["orientation"]) <= obj_idx: 
                    continue
                ori = cam["orientation"][obj_idx] 
                # update
                ori_delta = target_ori_delta[obj_idx]
                pseudo_azi = ori.azimuth + ori_delta.get("azimuth", 0)
                pseudo_pol = ori.polar + ori_delta.get("polar", 0)

                R_matrix = cam["rotation matrix"]
                # back to 3D
                world_f = self.project_orientation(pseudo_azi, pseudo_pol, R_matrix)
                # back to 3D then weighted
                front_list.append(world_f)
                score_list.append(each_score[cam_idx])

            weight = torch.tensor(score_list, dtype=torch.float32, device=self.device)
            front = torch.stack(front_list)
            weight = weight / weight.sum()
            weight_front = (front * weight[:, None]).sum(dim=0)
            weight_orient.append(weight_front)
        return torch.stack(weight_orient, dim=0)  # (N_obj, 3)

    def compute(self, camera_R, camera_T, default_ori=False, w_forward=2.0, w_point=2.0, w_gravity=2.0):
        centers = torch.stack([m["center"] for m in self.mask_index], dim=0).to(self.device)  # (N,3)
        obj_center = centers.mean(dim=0)  # (3,)

        R_c2w = camera_R
        R_w2c = camera_R.transpose(0, 1)
        cam_center = -camera_R @ camera_T

        z_cam = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        up_cam = torch.tensor([0.0, -1.0, 0.0], device=self.device) 

        if default_ori:
            world_up = torch.tensor([0.0, -1.0, 0.0], device=self.device)
            world_up_cam = F.normalize(R_w2c @ world_up, dim=0) 
            loss_gravity = 1.0 - F.cosine_similarity(up_cam.unsqueeze(0), world_up_cam.unsqueeze(0), dim=-1)
            loss = w_gravity * loss_gravity
        else:
            world_up = torch.tensor([0.0, -1.0, 0.0], device=self.device)
            world_up_cam = F.normalize(R_w2c @ world_up, dim=0) 
            loss_gravity = 1.0 - F.cosine_similarity(up_cam.unsqueeze(0), world_up_cam.unsqueeze(0), dim=-1)

            obj_ori_m = self.pseudo_orientation.detach() # (3,)
            # left-invariant forward 
            obj_ori_cam = F.normalize(R_w2c @ obj_ori_m, dim=0) 
            loss_forward = 1.0 - F.cosine_similarity(z_cam.unsqueeze(0), obj_ori_cam.unsqueeze(0), dim=-1)

            # left-invariant look-at
            point_obj = obj_center - cam_center 
            point_obj_cam = F.normalize(R_w2c @ point_obj, dim=0)
            loss_point = 1.0 - F.cosine_similarity(z_cam.unsqueeze(0), point_obj_cam.unsqueeze(0), dim=-1)

            loss = w_forward * loss_forward + w_point * loss_point + w_gravity * loss_gravity
        return loss


class LayoutLoss:
    def __init__(self, init_data, mllm_parser_layout, args, roll=None):
        self.device = args.device
        self.target_bbox_path = args.target_bbox_path
        # camera intrinsics info
        cam_info = init_data[0]
        self.width = cam_info["width"]
        self.height = cam_info["height"]

        xs = torch.linspace(0.0, 1.0, self.width, device=self.device)
        ys = torch.linspace(0.0, 1.0, self.height, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        self.grid_x = grid_x.unsqueeze(0)  # (1, H, W) 
        self.grid_y = grid_y.unsqueeze(0)  # (1, H, W)

        # from MLLM
        target_bboxes = []
        for item in mllm_parser_layout:
            x_min, y_min, x_max, y_max = item["layout"]
            target_bboxes.append([x_min, y_min, x_max, y_max])
        self.target_layout_range = torch.tensor(target_bboxes, dtype=torch.float32, device=self.device)
       
        if roll is not None:
            self.target_layout_range = self.rotate_layout(self.target_layout_range, self.height, self.width, roll)
            self.roll = True
            draw_target_bbox_obb(
                self.target_layout_range,
                self.height,
                self.width,
                self.target_bbox_path
            )
        else:
            self.roll = False
            draw_target_bbox_aabb(
                self.target_layout_range,
                self.height,
                self.width,
                self.target_bbox_path
            )
    
    def rotate_layout(self, layout, H, W, roll_angle):
        """
        layout: (N,4) normalized [x0,y0,x1,y1] in [0,1]
        H, W: image size
        angle_deg: same as img.rotate(angle_deg), expand=False
        return: rotated normalized AABB bboxes, clipped to [0,1]
        """

        # angle
        roll_angle = torch.tensor(roll_angle, device=self.device) 
        roll_angle = torch.clamp(roll_angle, -90.0, 90.0)
        gamma = torch.deg2rad(roll_angle)
        cos, sin = torch.cos(gamma), torch.sin(gamma)

        scale = torch.tensor([W - 1, H - 1], device=self.device)  # [sx, sy]
        center = scale / 2.0                                      # [(W-1)/2, (H-1)/2]
        # (N,4) -> (N,4,2) corners in pixel coords
        x0y0 = layout[:, [0, 1]]
        x1y1 = layout[:, [2, 3]]
        corners = torch.stack([
            x0y0,
            torch.stack([x1y1[:, 0], x0y0[:, 1]], dim=-1),
            x1y1,
            torch.stack([x0y0[:, 0], x1y1[:, 1]], dim=-1),
        ], dim=1) * scale  # (N,4,2)

        # rotate around image center
        p = corners - center
        x_rot = cos * p[..., 0] - sin * p[..., 1]
        y_rot = sin * p[..., 0] + cos * p[..., 1]
        corners_rot = torch.stack([x_rot, y_rot], dim=-1) + center  # (N,4,2)
        # back to normalized (N,4)
        layout_rot = corners_rot / scale
        return layout_rot

    def rotate_distance(self, corners: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, eps=1e-8):
        vi = corners  # (N,4,2)
        vj = vi.roll(shifts=-1, dims=1)
        # mask center is inside or not
        ex = vj[..., 0] - vi[..., 0]   # (N,4)
        ey = vj[..., 1] - vi[..., 1]
        rx = cx[:, None] - vi[..., 0]
        ry = cy[:, None] - vi[..., 1]
        cross = ex * ry - ey * rx      # (N,4)
        inside = (cross >= 0).all(dim=1) | (cross <= 0).all(dim=1)  # (N,)

        # distance for center not inside
        abx, aby = ex, ey # (N,4)
        apx, apy = rx, ry
        denom = abx * abx + aby * aby + eps
        t = (apx * abx + apy * aby) / denom
        t = t.clamp(0.0, 1.0)

        closest_x = vi[..., 0] + t * abx
        closest_y = vi[..., 1] + t * aby
        dx = cx[:, None] - closest_x
        dy = cy[:, None] - closest_y
        dist = torch.sqrt(dx * dx + dy * dy + eps)  # (N,4)
        dist_min = dist.min(dim=1).values 
        return torch.where(inside, torch.zeros_like(dist_min), dist_min)

    def rotate_mask(self, corners: torch.Tensor):
        """
        corners: (N,4,2) normalized corners (x,y)
        return: (N,H,W) float mask in {0,1}
        """
        x = corners[:, :, 0].view(-1, 4, 1, 1)
        y = corners[:, :, 1].view(-1, 4, 1, 1)

        # edges i -> j (j = i+1 mod 4)
        x_i, y_i = x, y
        x_j, y_j = x.roll(shifts=-1, dims=1), y.roll(shifts=-1, dims=1)

        # cross_z = (e x (p - vi))_z, where e = vj - vi
        cross = (x_j - x_i) * (self.grid_y - y_i) - (y_j - y_i) * (self.grid_x - x_i)  # (N,4,H,W)
        inside_mask = (cross >= 0).all(dim=1).float()
        return inside_mask

    def _ensure_grid(self, H, W, device, dtype):
        if (not hasattr(self, "grid_x")) or self.grid_x.shape[-2:] != (H, W) or self.grid_x.device != device:
            xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
            ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
            self.grid_x = grid_x.unsqueeze(0)  # (1, H, W)
            self.grid_y = grid_y.unsqueeze(0)  # (1, H, W)

    def mask_centroid(self, opacity: torch.Tensor):
        """
        input: opacity: (N_obj, H, W)
        return cx: (N_obj,) cy: (N_obj,)
        """

        if opacity.sum() < 1e-6:
            raise AssertionError("mask_center: opacity map contains no foreground (all zeros).")
        B, H, W = opacity.shape
        self._ensure_grid(H, W, opacity.device, opacity.dtype)

        mass = opacity.view(opacity.shape[0], -1).sum(dim=-1)  # (N_obj,)
        cx = (opacity * self.grid_x).view(opacity.shape[0], -1).sum(dim=-1) / mass
        cy = (opacity * self.grid_y).view(opacity.shape[0], -1).sum(dim=-1) / mass
        return cx, cy

    def layout_center_loss(self, opacity, bboxes, beta=0.1):
        """
        opacity: (N_obj, H, W)
        bboxes: (N_obj, 4)
        return: (N_obj,) loss
        """
        cx, cy = self.mask_centroid(opacity)
        
        if self.roll:
            dxy = self.rotate_distance(bboxes, cx, cy)
            zero = torch.zeros_like(dxy)
            loss_center = F.smooth_l1_loss(dxy, zero, beta=beta, reduction="none")
        else:
            x0, y0, x1, y1 = bboxes.unbind(dim=-1)

            dx_left  = torch.clamp(x0 - cx, min=0)
            dx_right = torch.clamp(cx - x1, min=0)
            dx = dx_left + dx_right  

            dy_top   = torch.clamp(y0 - cy, min=0)
            dy_bottom= torch.clamp(cy - y1, min=0)
            dy = dy_top + dy_bottom

            zero = torch.zeros_like(dx)
            loss_x = F.smooth_l1_loss(dx, zero, beta=beta, reduction="none")
            loss_y = F.smooth_l1_loss(dy, zero, beta=beta, reduction="none")
            loss_center = loss_x + loss_y
        return loss_center

    def layout_region_loss(self, opacity, bboxes, w_in=6.0, w_out=2.0):
        """
        opacity: (N_obj, H, W)
        bboxes: (N_obj, 4)
        """
        N, H, W = opacity.shape

        if self.roll:
            poly_mask = self.rotate_mask(bboxes)  # (N,H,W)
            inside = opacity * poly_mask
            outside = opacity * (1.0 - poly_mask)
        else:
            x0, y0, x1, y1 = [bboxes[:, i].view(N, 1, 1) for i in range(4)]
            
            box_mask = ((self.grid_x >= x0) & (self.grid_x <= x1) &
                        (self.grid_y >= y0) & (self.grid_y <= y1)).float()   # (N, H, W)
            inside = opacity * box_mask
            outside = opacity * (1.0 - box_mask)

        inside_mean = inside.view(N, -1).mean(dim=-1)    # (N,)
        outside_mean = outside.view(N, -1).mean(dim=-1)  # (N,)

        loss_inside = 1.0 - inside_mean  
        loss_outside = outside_mean     
        loss_region = w_in * loss_inside + w_out * loss_outside # (N,)

        return loss_region


    def compute(self, opacity, w_center=1.0, w_region=1.0, reduce="mean"):
        """
        opacity: (N_obj, H, W)
        reduce: "sum" / "mean"
        """
        assert opacity.dim() == 3, "expect opacity.shape = (N_obj, H, W)"
        N_obj = opacity.shape[0]
        assert N_obj == self.target_layout_range.shape[0], \
            f"Object number mismatch: opacity has {N_obj}, target bboxes have {self.target_layout_range.shape[0]}"

        bboxes = self.target_layout_range  # (N_obj, 4)/(N_obj, 4, 2)
        loss_center = self.layout_center_loss(opacity, bboxes)   # (N_obj,)
        loss_region = self.layout_region_loss(opacity, bboxes)   # (N_obj,)

        loss_per_obj = w_center * loss_center + w_region * loss_region  # (N_obj,)

        if reduce == "sum":
            total_loss = loss_per_obj.sum()
        elif reduce == "mean":
            total_loss = loss_per_obj.mean()
        else:
            raise ValueError(f"Unknown reduce='{reduce}'")

        return total_loss
