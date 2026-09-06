import os
import sys
from argparse import ArgumentParser, Namespace
from CameraGS import ModelParams, PipelineParams, get_combined_args
from .default import GAUSSIANS_PATH, OUTPUT_PATH, USER_INPUT

def main_args(parser: ArgumentParser):
    group = parser.add_argument_group("main parameters for optimization")
    # input
    group.add_argument("--user_input", type=str, default="", help="text from user to describe expected frame")
    # default path
    group.add_argument("--render_path", default="", type=str, help="path of rendered images")
    group.add_argument("--camera_R_path", default="", type=str, help="path of camera rotation")
    group.add_argument("--camera_T_path", default="", type=str, help="path of camera translation")
    # optimize
    group.add_argument("--cam_rot_lr", default=0.007, type=float, help="learning rate of camera rotation delta")
    group.add_argument("--cam_trans_lr", default=0.005, type=float, help="learning rate of camera translation delta") 
    group.add_argument("--opt_steps", default=1500, type=int, help="optimization steps")
    group.add_argument("--converge", default=1e-4, type=float, help="threshold for stopping") # 1e-4 or 1e-5
    # loss
    group.add_argument("--alpha", default=1.0, type=float, help="weight of orient loss")
    group.add_argument("--beta", default=1.0, type=float, help="weight of layout loss")
    # output
    group.add_argument("--save_image_every", type=int, default=20, help="save image every n steps")
    return parser

def mllm_args(parser:ArgumentParser):
    group = parser.add_argument_group("main parameters for mllm")
    group.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="model id for mllm")
    group.add_argument("--model_root", type=str, default="./fg_clip_base", help="model root for rough selection in stage 1")
    group.add_argument("--batch_size", default=32, type=int, help="image batch size for inference")
    # stage 1 filter
    group.add_argument("--score_topk", default=2, type=int, help="maximum images")
    group.add_argument("--score_threshold", default=42.0, type=float, help="threshold for filtering images") 
    group.add_argument("--filter_fgclip", default=50, type=int, help="top images in FG-CLIP for filter again")
    # stage 2
    group.add_argument("--n_neighbors", default=50, type=int, help="maximum number of neighbors in KNN")
    group.add_argument("--percentile_threshold", default=95.0, type=float, help="threshold for filtering points")
    group.add_argument("--camera_vertical", action="store_true", help="if the camera up direction is vertical to the ground") 
    return parser

def get_args_parser():
    parser = ArgumentParser(description="optimization")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)

    # other args
    parser = main_args(parser)
    parser = mllm_args(parser)
    
    # file path
    parser.set_defaults(
        **GAUSSIANS_PATH,
        **OUTPUT_PATH,
        **USER_INPUT
    )
    
    args = get_combined_args(parser)
    args.model = mp.extract(args)
    args.pipeline = pp.extract(args)

    args.camera_path = os.path.join(args.model.model_path, "cameras.json")
    args.device = "cuda"

    return args