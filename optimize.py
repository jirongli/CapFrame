from CameraGS import (
    GaussianModel, render,
    CustomCam,
    focal2fov, 
    searchForMaxIteration,
    update_pose
)
from utils import load_topk_cameras, draw_opacity_map
from loss import OrientationLoss, LayoutLoss
from rendering import Renderer
from configs.arguments import get_args_parser
from mllm import MLLM
from GroundedSAM.inference import GroundedSAM
from OrientAnything.inference import OrientAnything

import torch
import torchvision
import json
import os
import sys
import cv2
import random
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as Transforms
from argparse import ArgumentParser
from torch import optim
from tqdm import tqdm
from PIL import Image

class Optimizer():
    def __init__(self, init_cameras, gaussian_mask, gaussians, mllm_output, args):
        self.device = args.device
        self.pipeline = args.pipeline
        self.default_ori = args.camera_vertical
        self.converge = args.converge
        # set cameras
        # starting point of GD
        max_score = max(cam["score"] for cam in init_cameras)
        best_indices = [
            idx for idx, cam in enumerate(init_cameras)
            if cam["score"] == max_score
        ]
        self.idx = random.choice(best_indices)
        info = init_cameras[self.idx]
        self.fx = info['fx']
        self.fy = info['fy']
        self.width = info['width']
        self.height = info['height']
        self.cx = info["width"] / 2.0
        self.cy = info["height"] / 2.0
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        cameras = []
        for cam in init_cameras:
            R_mat = cam['rotation matrix']
            T_vec = cam['translation vector']
            c_id = cam['id']
            # initial setting
            viewpoint = CustomCam(self.width, self.height, self.fovy, self.fovx, R_mat, T_vec, c_id, "optimize")
            cameras.append(viewpoint)
        self.cameras = cameras
        # set scene
        self.gaussians = gaussians
        self.bg_color = [1, 1, 1] if args.model.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device=args.device)
        # object indices
        self.gaussian_mask = gaussian_mask

        # set loss
        self.alpha = args.alpha
        self.alpha_last = self.alpha / 2.0
        self.beta = args.beta
        if "orient" in mllm_output:
            mllm_parser_orient = mllm_output["orient"]
            self.ori_loss = OrientationLoss(init_cameras, gaussian_mask, mllm_parser_orient, args)
            # roll
            cam_roll = any(abs(d["orient"][-1]) > 1e-5 for d in mllm_parser_orient)
            if cam_roll:
                cam_rotation = float(mllm_parser_orient[0]["orient"][-1])
            else:
                cam_rotation = None
        elif self.default_ori:
            self.ori_loss = OrientationLoss(init_cameras, gaussian_mask, None, args)
            cam_rotation = None
        else:
            self.ori_loss = None
            cam_rotation = None
        mllm_parser_layout = mllm_output["layout"]
        self.lay_loss = LayoutLoss(init_cameras, mllm_parser_layout, args, cam_rotation)
        # save output
        self.save_image_every = args.save_image_every
        self.render_path = args.render_path
        self.camera_R_path = args.camera_R_path
        self.camera_T_path = args.camera_T_path

    def run(self):
        R_list = []
        T_list = []
        ori_loss_list = []
        lay_loss_list = []

        viewpoint = self.cameras[self.idx]
        # set optimizer
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": args.cam_rot_lr,
                "name": "rot_from_{}".format(viewpoint.camera_id),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": args.cam_trans_lr,
                "name": "trans_from_{}".format(viewpoint.camera_id),
            }
        )
        opt = optim.Adam(opt_params)

        with tqdm(total=args.opt_steps) as pbar:
            for i in range(args.opt_steps):
                # camera update
                new_R, new_T = update_pose(viewpoint)
                # render
                img_render = render(viewpoint, self.gaussians, self.pipeline, self.background)
                rgb_rendering = img_render["render"]
                img_rgb = rgb_rendering.unsqueeze(0)
                # mask
                opacity_map = []
                # depth_obj = []
                for obj_id, mask in enumerate(self.gaussian_mask):
                    mask_render = render(viewpoint, self.gaussians, self.pipeline, self.background, mask["index"])
                    img_opa = mask_render["alpha"].squeeze(0)
                    opacity_map.append(img_opa)

                opacity_map = torch.stack(opacity_map, dim=0)
                if (i+1) % self.save_image_every == 0:
                        all_obj_opa = draw_opacity_map(opacity_map)
                        filename_alpha = f"alpha_iteration{i}.png"
                        # torchvision.utils.save_image(all_obj_opa, os.path.join(self.render_path, filename_alpha))

                # compute loss
                progress = min(args.opt_steps / (i+1), 1.0)
                alpha = self.alpha * (1 - progress) + self.alpha_last * progress
                if self.default_ori:
                    ori_loss = self.ori_loss.compute(new_R, new_T, default_ori=True)
                    alpha = self.alpha # Keep full value when only gravity applies
                elif self.ori_loss is None:
                    ori_loss = torch.tensor(0.0, device=self.device)
                else:
                    ori_loss = self.ori_loss.compute(new_R, new_T)
                lay_loss = self.lay_loss.compute(opacity_map)
                loss = alpha * ori_loss + self.beta * lay_loss
                ori_loss_list.append(ori_loss.item())
                lay_loss_list.append(lay_loss.item())

                # compute backward pass
                loss.backward()
                opt.step()
                opt.zero_grad()
            
                if (i+1) % self.save_image_every == 0:
                    filename_rgb = f"rgb_iteration{i}.png"
                    torchvision.utils.save_image(rgb_rendering, os.path.join(self.render_path, filename_rgb))
                    # save camera parameters
                    R_array = viewpoint.R.detach().cpu().numpy()
                    T_array = viewpoint.T.detach().cpu().numpy()
                    R_list.append(R_array)
                    T_list.append(T_array)
                pbar.set_description("Orient Loss: {:.3f} Layout Loss: {:.3f}".format(ori_loss.item(), lay_loss.item()))
                pbar.update(1)

                cur_loss = loss.item()
                if i == 0:
                    filename_rgb = f"rgb_iteration{i}.png"
                    os.makedirs(self.render_path, exist_ok=True)
                    torchvision.utils.save_image(rgb_rendering, os.path.join(self.render_path, filename_rgb))
                else: 
                    converged = abs(cur_loss - prev_loss)
                    if converged < self.converge:
                        print("End of Optimization.")
                        filename_rgb = f"rgb_iteration{i}.png"
                        torchvision.utils.save_image(rgb_rendering, os.path.join(self.render_path, filename_rgb))
                        break
                prev_loss = cur_loss
            np.save(self.camera_R_path, R_list)
            np.save(self.camera_T_path, T_list)

if __name__ == "__main__":
    
    args = get_args_parser()

    # input text
    input_text = args.user_input 

    # load Gaussian environment
    loaded_iter = searchForMaxIteration(os.path.join(args.model.model_path, "point_cloud"))
    gaussians = GaussianModel(args.model.sh_degree)
    gaussians.load_ply(os.path.join(args.model.model_path, "point_cloud", f'iteration_{loaded_iter}', "point_cloud.ply"))

    # load results from stage1
    mllm_parser = MLLM(args, score_topk=args.score_topk, score_threshold=args.score_threshold)
    questions = mllm_parser.TextQuestioner(input_text)
    cam_topk, cam_score_topk = mllm_parser.ImageEvaluator(questions)

    # initial camera and data setting (top-k: id, score, img_tensor)
    data = load_topk_cameras(cam_topk, cam_score_topk, args.camera_path, args.model.source_path, args.device)
    # render depth map
    Renderer = Renderer(data, gaussians, args)                                                      
    data = Renderer.render_depth(data)

    # text parse from input text 
    mllm_output = {}
    object_texts = mllm_parser.ObjectParser(data)
    # GroundedSAM
    mllm_parser_layout = mllm_parser.LayoutParser(object_texts)
    GroundedSAM = GroundedSAM(object_texts, args.device)
    data = GroundedSAM.infer(data)
    mllm_output["layout"] = mllm_parser_layout

    # get mask indices
    gaussian_mask = Renderer.get_mask_index(data, args)

    # Orient Anything 
    mllm_parser_orient = mllm_parser.OrientParser(object_texts)
    if mllm_parser_orient != -1:
        OrientAny = OrientAnything(args)
        data = OrientAny.infer(data, mllm_parser_orient, gaussian_mask)
        mllm_output["orient"] = mllm_parser_orient
        args.camera_vertical = False

    # optimize
    optimizer = Optimizer(data, gaussian_mask, gaussians, mllm_output, args)
    optimizer.run()
