import os

import cv2
import numpy as np
from typing import Any, List, Dict, Union, Tuple
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from GroundedSAM.utils import refine_masks, get_boxes, DetectionResult

class GroundedSAM():
    def __init__(self, object_texts, device, threshold=0.3, polygon_refinement=True,
                 detector_id="IDEA-Research/grounding-dino-base", segmenter_id="facebook/sam-vit-base"):
        self.device = device
        self.object_texts = object_texts
        self.threshold = threshold
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.polygon_refinement = polygon_refinement
    
    def detect(
        self,
        image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        object_detector = pipeline(model=self.detector_id, task="zero-shot-object-detection", device=self.device)

        labels = [label if label.endswith(".") else label+"." for label in self.object_texts]

        results = object_detector(image, candidate_labels=labels, threshold=self.threshold)
        # sort by the order of input_labels
        label_to_index = {lab: idx for idx, lab in enumerate(labels)}
        results = sorted(results, key=lambda r: label_to_index.get(r["label"], 9999))

        results = [DetectionResult.from_dict(result) for result in results]
        # top-1 filter
        best = {}
        for r in results:
            lab = r.label
            # keep highest one
            if lab not in best or r.score > best[lab].score:
                best[lab] = r
        results_f = list(best.values())
        return results_f

    def segment(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]]
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """
        segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self.device)
        processor = AutoProcessor.from_pretrained(self.segmenter_id)

        boxes = get_boxes(detection_results)
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = segmentator(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, self.polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask
            # mask center
            mask_bin = (mask > 0).astype(np.uint8)
            image_moments = cv2.moments(mask_bin)
            if image_moments["m00"] != 0:
                cx = image_moments["m10"] / image_moments["m00"]
                cy = image_moments["m01"] / image_moments["m00"]
            else:
                cx, cy = None, None
            detection_result.mask_center = (cx, cy)

        return detection_results

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str]
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = load_image(image)

        detections = self.detect(image)
        detections = self.segment(image, detections)

        return np.array(image), detections

    def mask_crop_image(self, item): 
        image = Image.open(item["image path"]).convert("RGB")
        cam_id = item["id"]
        processed_image = []
        text_id = 0
        os.makedirs("./camera_result/check/cropped_images", exist_ok=True)
        for detect in item["detection"]:
            label = detect.label
            name = label[:-1] if label.endswith('.') else label
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
                crop_img.save(f"./camera_result/check/cropped_images/cam{cam_id}_obj{text_id}.jpg")
                text_id += 1
            item["cropped image"] = processed_image
        return processed_image
    
    def infer(self, data):
        for item in data:
            img = Image.open(item["image path"]).convert("RGB")
            image_array, detections = self.grounded_segmentation(img)
            item["image"] = image_array
            item["detection"] = detections
            crop_images = self.mask_crop_image(item) 
        return data
