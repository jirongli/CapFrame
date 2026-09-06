import json
import torch
import cv2
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from PIL import Image

def to_device(data):
    return 1

def load_topk_cameras(topk_id:List[int], topk_score:List[int], json_path: str, img_path: str, device):
    with open(json_path, "r") as f:
        data = json.load(f)

    selected_cameras = []
    for cam in data:
        if cam.get("id") in topk_id:
            # get image path
            img_dir = Path(img_path) / cam["img_name"]
            # get scores
            id_index = topk_id.index(cam["id"]) 
            score = topk_score[id_index]
            # load as array
            R = np.array(cam["rotation"], dtype=np.float32)
            camera_center = np.array(cam["position"], dtype=np.float32) 
            t = -R.T @ camera_center 
            # to tensor
            R = torch.from_numpy(R).to(device)    
            t = torch.from_numpy(t.astype(np.float32)).to(device)    
            selected_cameras.append({
                "id": cam["id"],
                "image path": img_dir,
                "score": score,
                "rotation matrix": R,
                "translation vector": t,
                "width": cam["width"],
                "height": cam["height"],
                "fx": cam["fx"],
                "fy": cam["fy"],
            })
    
    if not selected_cameras:
        raise ValueError(f"No cameras found for ids {topk_id}")
    return selected_cameras

def resize_max_side(img: Image.Image, max_side: int = 1000) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize(
        (int(w * scale), int(h * scale)),
        Image.BICUBIC
    )

def darken_color(rgb, factor=0.6):
    """
    rgb: (r, g, b) in [0,1]
    factor < 1: darker, e.g. 0.6~0.8
    """
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0, min(1, l * factor))
    s = min(1, s * 1.1)   
    return colorsys.hls_to_rgb(h, l, s)

def draw_target_bbox_aabb(bboxes, H, W, save_path, thickness=3):
    """
    bboxes: (N, 4) tensor, [x_min, y_min, x_max, y_max], normalized
    H, W: image height/width
    save_path: output PNG path
    """
    if hasattr(bboxes, "detach"):
        bboxes = bboxes.detach().cpu().numpy()

    N = bboxes.shape[0]
    canvas = np.ones((H, W, 3), dtype=np.float32) * 0.25  

    cmap = plt.get_cmap("Set3")   

    for i in range(N):
        color = darken_color(cmap(i % cmap.N)[:3])

        x0, y0, x1, y1 = bboxes[i].tolist()

        x0 = int(x0 * W)
        x1 = int(x1 * W)
        y0 = int(y0 * H)
        y1 = int(y1 * H)

        t = thickness

        # top & bottom
        canvas[max(0, y0 - t):min(H, y0 + t), x0:x1] = color
        canvas[max(0, y1 - t):min(H, y1 + t), x0:x1] = color

        # left & right
        canvas[y0:y1, max(0, x0 - t):min(W, x0 + t)] = color
        canvas[y0:y1, max(0, x1 - t):min(W, x1 + t)] = color

    plt.imsave(save_path, canvas)

def draw_thick_point_obb(x, y, H, W, canvas, color, radius_thickness):
    x0 = max(0, x - radius_thickness)
    x1 = min(W, x + radius_thickness + 1)
    y0 = max(0, y - radius_thickness)
    y1 = min(H, y + radius_thickness + 1)
    canvas[y0:y1, x0:x1] = color

def draw_target_bbox_obb(corners, H, W, save_path, radius_thickness=2):
    if hasattr(corners, "detach"):
        corners = corners.detach().cpu().numpy()

    N = corners.shape[0]
    canvas = np.ones((H, W, 3), dtype=np.float32) * 0.25
    cmap = plt.get_cmap("Set3")

    for i in range(N):
        color = darken_color(cmap(i % cmap.N)[:3])
        # normalized -> pixel coords
        pts = corners[i].astype(np.float32).copy()
        pts[:, 0] *= (W - 1)
        pts[:, 1] *= (H - 1)
        # edges: (0->1), (1->2), (2->3), (3->0)
        for k in range(4):
            xA, yA = pts[k]
            xB, yB = pts[(k + 1) % 4]

            dx = xB - xA
            dy = yB - yA
            length = int(max(abs(dx), abs(dy))) + 1 
            if length <= 1:
                draw_thick_point_obb(int(round(xA)), int(round(yA)), H, W, canvas, color, radius_thickness)
                continue

            xs = np.linspace(xA, xB, length)
            ys = np.linspace(yA, yB, length)
            for x, y in zip(xs, ys):
                draw_thick_point_obb(int(round(x)), int(round(y)), H, W, canvas, color, radius_thickness)

    plt.imsave(save_path, canvas)

def draw_opacity_map(opacity_map):
    """
    opacity_map: (N_obj, H, W)
    return: (3, H, W) RGB
    """
    N, H, W = opacity_map.shape
    canvas = torch.ones(3, H, W, device=opacity_map.device) * 0.25

    cmap = plt.get_cmap("Set3") 
    for i in range(N):
        color = darken_color(cmap(i % cmap.N)[:3])
        color = torch.tensor(color, device=opacity_map.device).view(3, 1, 1)
        canvas += opacity_map[i].unsqueeze(0) * color 
    canvas = canvas.clamp(0, 1)
    return canvas