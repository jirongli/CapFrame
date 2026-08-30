import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download

from OrientAnything.Orient_Anything.paths import *
from OrientAnything.Orient_Anything.vision_tower import DINOv2_MLP
from OrientAnything.Orient_Anything.utils import *
from OrientAnything.Orient_Anything.inference import *
from OrientAnything.utils import OrientResult

class OrientAnything():
    def __init__(self, args, cache_dir="./OrientAnything/", 
                processed_image_dir="./OrientAnything/cropped_images/",
                do_rm_bkg=True, do_infer_aug=False):
        self.device = args.device
        self.processed_image_dir = args.processed_image_dir
        self.cache_dir = cache_dir
        self.do_rm_bkg = do_rm_bkg 
        self.do_infer_aug = do_infer_aug 

    def mask_crop_image(self, item, obj_has_ori):
        if not os.path.exists(self.processed_image_dir):
            os.makedirs(self.processed_image_dir)

        image = Image.open(item["image path"]).convert("RGB")
        cam_id = item["id"]
        processed_image = []
        text_id = 0
        for detect in item["detection"]:
            label = detect.label
            name = label[:-1] if label.endswith('.') else label
            if name not in obj_has_ori:
                continue
            score = detect.score
            box = detect.box
            mask = detect.mask

            if mask is not None:
                # Convert mask to uint8
                mask_uint8 = mask.astype(np.uint8)
                mask_pil = Image.fromarray(mask_uint8)
                img_tem = image.copy()
                img_tem.putalpha(mask_pil)  
                # white background
                white_bg = Image.new("RGB", img_tem.size, (255, 255, 255))
                white_bg.paste(img_tem, mask=mask_pil)
                # crop
                xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
                crop_img = white_bg.crop((xmin, ymin, xmax, ymax))
                processed_image.append(crop_img)
                crop_img.save(f"{self.processed_image_dir}cam{cam_id}_obj{text_id}.jpg")
                text_id += 1
            item["cropped image"] = processed_image
        return processed_image

    def infer(self, data, object_name, valid_mask):
        obj_has_ori = [o["object"] for o in object_name]
        for item in data: 
            crop_images = self.mask_crop_image(item, obj_has_ori)

            ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="ronormsigma1/dino_weight.pt", repo_type="model", cache_dir=self.cache_dir, resume_download=True)
            dino = DINOv2_MLP(dino_mode = 'large', in_dim = 1024, out_dim = 360+180+360+2,
                                evaluate = True, mask_dino = False, frozen_back = False
            )
            dino.eval()
            dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            dino = dino.to(self.device)
            val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir=self.cache_dir)

            results = [] 
            for img in crop_images:
                if self.do_infer_aug:
                    rm_bkg_img = background_preprocess(img, True)
                    angles = get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, self.device)
                else:
                    rm_bkg_img = background_preprocess(img, self.do_rm_bkg)
                    angles = get_3angle(rm_bkg_img, dino, val_preprocess, self.device)
                azimuth = float(angles[0])
                polar = float(angles[1])
                rotation = float(angles[2])
                confidence = float(angles[3])
                results.append({
                    "azimuth": azimuth,
                    "polar": polar,
                    "rotation": rotation,
                    "confidence": confidence
                })
            results = [OrientResult.from_dict(result) for result in results]
            item["orientation"] = results
        return data

