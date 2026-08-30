import os
import re
import json
import torch
import random
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

from utils import resize_max_side
from prompts import prompt_generate_questions, prompt_score_images, prompt_recognize_objects, prompt_parse_orient, prompt_parse_layout

class MLLM:
    def __init__(self, args,  tokenize=False, add_generation_prompt=True, max_new_tokens_short=268, do_sample=False, temperature=0.7,
    score_threshold=90.0, score_topk=3, padding=True, return_tensors="pt", skip_special_tokens=True, clean_up_tokenization_spaces=False, trust_remote_code=True, use_fast=True):
        self.device = args.device
        self.model_id = args.model_id
        self.batch_size = args.batch_size
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.user_prompt = args.user_input

        # model parameters
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.max_new_tokens_short = max_new_tokens_short
        self.do_sample = do_sample
        self.temperature = temperature
        self.padding = padding
        self.return_tensors = return_tensors
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces

        self.trust_remote_code = trust_remote_code
        self.use_fast = use_fast
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"

        # stage 1 model FG_CLIP
        self.model_root = args.model_root
        self.fg_model = AutoModelForCausalLM.from_pretrained(self.model_root,trust_remote_code=self.trust_remote_code).cuda()
        self.fg_tokenizer = AutoTokenizer.from_pretrained(self.model_root)
        self.fg_image_processor = AutoImageProcessor.from_pretrained(self.model_root, use_fast=self.use_fast)
        self.camera_path = args.camera_path
        self.source_path = args.source_path
        self.text_prompt = args.user_input
        self.score_threshold = score_threshold
        self.score_topk = score_topk
        self.filter_fgclip = getattr(args, "filter_fgclip", None)

    def _load_json(self, s: str):
        s = s.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return [obj]
            return obj
        except json.JSONDecodeError:
            pass

        objs = []
        depth = 0
        start = None
        for i, ch in enumerate(s):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(s[start:i+1])
                    start = None

        if not objs:
            raise ValueError("LLM output format error: not valid JSON")
        return json.loads('[' + ','.join(objs) + ']')

    def init_selection(self):
        with open(self.camera_path, "r") as f:
            cameras = json.load(f)
        cameras_sorted = sorted(cameras, key=lambda x: x["id"])
        image_paths = [
            os.path.join(self.source_path, item["img_name"])
            for item in cameras_sorted
        ]
        images = [Image.open(p).convert("RGB") for p in image_paths]

        top_k = max(1, len(images)//8) 
        image_size = 224
        resized_images = [
            im.resize((image_size, image_size)) for im in images
        ]
        with torch.no_grad():
            image_inputs = self.fg_image_processor(
                images=resized_images,
                return_tensors=self.return_tensors
            )["pixel_values"].to(self.device)   # [B, 3, 224, 224]
            img_feats = self.fg_model.get_image_features(image_inputs)  # [B, D]
            img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
            prompt_inputs = self.fg_tokenizer(
                self.text_prompt,
                return_tensors=self.return_tensors,
                padding="max_length",
                truncation=True,
                max_length=77
            ).input_ids.to(self.device)
            txt_feat = self.fg_model.get_text_features(prompt_inputs, walk_short_pos=True)  # [1, D]
            txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
            logits = img_feats @ txt_feat.T 
            logits = self.fg_model.logit_scale.exp() * logits
        scores = logits[:, 0]
        cam_ids = torch.argsort(scores, descending=True)[:top_k].tolist()
        selected_img_paths = [image_paths[i] for i in cam_ids]
        return selected_img_paths, cam_ids

    def chunks(self, samples, n):
        for i in range(0, len(samples), n):
            yield samples[i:i + n]
    
    def adaptive_filter(self, sorted_indices, sorted_scores, half_topk_cam, score_threshold, score_k):
        m = sum(s > score_threshold for s in sorted_scores)
        n = len(sorted_scores)
        if m <= 1:
            assert m > 0, "No candidates left. Score threshold may be too high."
            cut = min(score_k, m)
            return sorted_indices[:cut], sorted_scores[:cut]
        
        gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(m - 1)]
        cut = int(np.argmax(gaps)) + 1
        if cut <= score_k:
            return sorted_indices[:cut], sorted_scores[:cut]
        # cut > score_k
        kth_score = sorted_scores[score_k - 1]
        # all scores > kth_score 
        higher = [(i, s) for i, s in enumerate(sorted_scores[:cut]) if s > kth_score]
        # all scores == kth_score 
        equal  = [(i, s) for i, s in enumerate(sorted_scores[:cut]) if s == kth_score]
        selected = list(higher)
        remain = score_k - len(selected)

        if remain > 0:
            # ramdom selection
            sampled = random.sample(equal, k=min(remain, len(equal)))
            selected.extend(sampled)

        # recover
        selected = sorted(selected, key=lambda x: x[0])
        final_indices = [sorted_indices[i] for i, _ in selected]
        final_scores  = [sorted_scores[i]  for i, _ in selected]

        # filter by FG-CLIP
        top_cam_set = set(half_topk_cam)
        filtered = [
            (idx, score)
            for idx, score in zip(final_indices, final_scores)
            if idx in top_cam_set
        ]
        assert len(filtered) > 0, ("All images are filtered out by FG-CLIP. Consider increasing top_k.")
        final_indices, final_scores = zip(*filtered)
        final_indices = list(final_indices)
        final_scores = list(final_scores)
        return final_indices, final_scores

    def _load_score(self, texts, topk_cam, q_length):
        all_means = []
        all_scores = []
        for i, text in enumerate(texts):
            scores = [int(s) for s in re.findall(r"Q\d+:\s*(\d+)", text)]
            mean_score = float(np.mean(scores))
            if scores is None:
                raise ValueError(f"No scores are extracted.")
            if len(scores) != q_length:
                raise ValueError(
                    f"[Score Extraction Error] Item {i}: "
                    f"expected {q_length} scores, got {len(scores)}.\n"
                )
            all_scores.append(scores)
            all_means.append(mean_score)
        sorted_pairs = sorted(enumerate(all_means), key=lambda x: x[1], reverse=True)
        sorted_indices = [topk_cam[i] for i, _ in sorted_pairs]
        sorted_scores  = [score for _, score in sorted_pairs]
        # filter by interval
        if self.filter_fgclip is None:
            self.filter_fgclip = len(topk_cam)//2
        half_topk_cam = topk_cam[:self.filter_fgclip]
        filtered_indices, filtered_scores = self.adaptive_filter(sorted_indices, sorted_scores, half_topk_cam, self.score_threshold, self.score_topk)
        return filtered_indices, filtered_scores

    def TextQuestioner(self, user_input):
        # llm process
        message = [
            {"role": "system", "content": [{"type": "text", "text": prompt_generate_questions}]},
            {"role": "user", "content": [{"type": "text", "text": user_input}]}
        ]
        texts = self.processor.apply_chat_template(message, tokenize=self.tokenize, add_generation_prompt=self.add_generation_prompt)

        inputs = self.processor(
            text=texts,
            padding=self.padding,
            return_tensors=self.return_tensors,
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens_short, do_sample=self.do_sample, temperature=None if not self.do_sample else self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_prompt = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=self.skip_special_tokens, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )[0]
        # parse result
        questions = self._load_json(output_text_prompt)
        return questions

    def ImageEvaluator(self, questions):
        questions_text = "\n".join(
            [f"Q{i+1}: {q}" for i, q in enumerate(questions)]
        )
        prompt_score_images_q = prompt_score_images.replace("{questions}", questions_text)
        topk_image_paths, topk_cam = self.init_selection()
        # batch inference
        all_outputs = []
        for path in self.chunks(topk_image_paths, self.batch_size):
            imgs = []
            for p in path:
                img_o = Image.open(p).convert("RGB")
                img_r = resize_max_side(img_o, max_side=1000)
                imgs.append(img_r)

            messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": prompt_score_images_q}]},
                {"role": "user", "content": [{"type": "image", "image": img}]},
            ]
            for img in imgs
            ]     
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ] 
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                padding=self.padding,
                return_tensors=self.return_tensors,
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens_short, do_sample=self.do_sample, temperature=None if not self.do_sample else self.temperature)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            all_outputs.extend(output_texts)
        # parse results
        cam_ids, cam_scores = self._load_score(all_outputs, topk_cam, len(questions))
        return cam_ids, cam_scores
    
    def ObjectParser(self, data):
        user_content = []
        for item in data:
            img_path = item["image path"]
            img = Image.open(img_path).convert("RGB")
            user_content.append({
                "type": "image",
                "image": img
            })
        user_content.append({
            "type": "text",
            "text": self.user_prompt
        })
        # llm process
        message = [
            {"role": "system", "content": [{"type": "text", "text": prompt_recognize_objects}]},
            {"role": "user", "content": user_content}
        ]
        texts = self.processor.apply_chat_template(message, tokenize=self.tokenize, add_generation_prompt=self.add_generation_prompt)
        image_inputs, _ = process_vision_info(message)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=self.padding,
            return_tensors=self.return_tensors,
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens_short, do_sample=self.do_sample, temperature=None if not self.do_sample else self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_prompt = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=self.skip_special_tokens, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )[0]
        # parse result
        object_texts = self._load_json(output_text_prompt)
        return object_texts

    def OrientParser(self, object_list):
        text_objects = {
            "text": self.user_prompt,
            "objects": object_list
        }
        user_content = json.dumps(text_objects, ensure_ascii=False)
        # llm process
        message = [
            {"role": "system", "content": [{"type": "text", "text": prompt_parse_orient}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]}
        ]
   
        texts = self.processor.apply_chat_template(message, tokenize=self.tokenize, add_generation_prompt=self.add_generation_prompt)

        inputs = self.processor(
            text=texts,
            padding=self.padding,
            return_tensors=self.return_tensors,
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens_short, do_sample=self.do_sample, temperature=None if not self.do_sample else self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_prompt = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=self.skip_special_tokens, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )[0]
        # parse result
        orient_texts = self._load_json(output_text_prompt)
        filtered = [obj for obj in orient_texts if obj["orient"] != [-1, -1, -1]]
        if len(filtered) == 0:
            orient = -1
        else:
            orient = filtered
        return orient

    def LayoutParser(self, object_list):
        text_objects = {
            "text": self.user_prompt,
            "objects": object_list
        }
        user_content = json.dumps(text_objects, ensure_ascii=False)
        # llm process
        message = [
            {"role": "system", "content": [{"type": "text", "text": prompt_parse_layout}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]}
        ]
   
        texts = self.processor.apply_chat_template(message, tokenize=self.tokenize, add_generation_prompt=self.add_generation_prompt)

        inputs = self.processor(
            text=texts,
            padding=self.padding,
            return_tensors=self.return_tensors,
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens_short, do_sample=self.do_sample, temperature=None if not self.do_sample else self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_prompt = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=self.skip_special_tokens, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )[0]
        # parse result
        layout_texts = self._load_json(output_text_prompt)
        return layout_texts