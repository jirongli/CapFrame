from CameraGS import (
    render,
    CustomCam,
    focal2fov
)
import os
import cv2
import torch
import numpy as np
import cupy as cp
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from cuml.neighbors import NearestNeighbors

class Renderer:
    def __init__(self, data, gaussians, args):
        self.device = args.device
        self.pipeline = args.pipeline
        # set scene
        self.gaussians = gaussians
        self.bg_color = [1, 1, 1] if args.model.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device=args.device)
        # set cameras
        info = data[0]

        self.fx = info['fx']
        self.fy = info['fy']
        self.orig_width = info['width']
        self.orig_height = info['height']
        self.cx = self.orig_width / 2.0
        self.cy = self.orig_height / 2.0
        self.fovx = focal2fov(self.fx, self.orig_width)
        self.fovy = focal2fov(self.fy, self.orig_height)

        cameras = []
        for cam in data:
            R_mat = cam['rotation matrix']
            T_vec = cam['translation vector']
            c_id = cam['id']
            # initial setting
            viewpoint = CustomCam(self.orig_width, self.orig_height, self.fovy, self.fovx, R_mat, T_vec, c_id, "test")
            cameras.append(viewpoint)
        self.cameras = cameras

    def project_to_world(self, obj_u, obj_v, camera):
        depth = camera["depth"]
        R = camera["rotation matrix"]
        T = camera["translation vector"]
        if obj_u is None or obj_v is None:
            depth_value = None
        else:
            depth_value = depth[obj_v.long(), obj_u.long()] 
        # to camera
        x_c = (obj_u - self.cx) * depth_value / self.fx
        y_c = (obj_v - self.cy) * depth_value / self.fy
        z_c = depth_value
        cam_coor = torch.stack([x_c, y_c, z_c], dim=0)
        # to world
        world_coor = R @ (cam_coor - T[:, None]) 
        return world_coor
        
    def render_depth(self, data):
        id_to_idx = {item["id"]: idx for idx, item in enumerate(data)}
        for view in self.cameras:
            img_render = render(view, self.gaussians, self.pipeline, self.background)
            img_depth = img_render["depth"][0]
            cam_id = view.camera_id
            if cam_id in id_to_idx:
                idx = id_to_idx[cam_id]
                data[idx]["depth"] = img_depth
        return data

    def get_mask_index(self, data, args):
        # gaussian points
        xyz_feat = self.gaussians._xyz.detach()

        # Select the largest mask and unproject all its pixels to 3D
        max_num_objs = max(len(cam["detection"]) for cam in data)
        mask_index_info = []
        large_mask = [] 
        for obj_idx in range(max_num_objs):
            count = 0.0
            for cam_idx, cam in enumerate(data):
                detect = cam["detection"][obj_idx] 
                obj_mask = detect.mask
                nonzero_count = np.count_nonzero(obj_mask)
                if nonzero_count > count:
                    large_cam = cam
                    count = nonzero_count
            large_mask.append(large_cam)

        for obj_id, cam in enumerate(large_mask):
            obj_mask = cam["detection"][obj_id].mask
            mask_t  = torch.from_numpy(obj_mask).to(device=self.device)   
            idx = torch.nonzero(mask_t > 0 , as_tuple=False)   # (N, 2)
            v = idx[:, 0].float()   
            u = idx[:, 1].float()
            mask_world = self.project_to_world(u, v, cam)

            # cuml
            query = mask_world.transpose(0, 1).detach()
            knn = NearestNeighbors(n_neighbors=args.n_neighbors)
            knn.fit(xyz_feat)
            distances, indices = knn.kneighbors(query)
            # 95% percentile
            threshold = cp.percentile(distances, args.percentile_threshold) 
            mask = distances <= threshold            
            # knn
            neighbors = cp.unique(indices[mask])  
            neighbors_np = cp.asnumpy(neighbors)
            neighbors = torch.from_numpy(neighbors_np).to(xyz_feat.device)
            local_region = xyz_feat[neighbors] 
            
            center = local_region.mean(axis=0)
            mask_index_info.append({
                "index": neighbors,
                "center": center
            })
        return mask_index_info
