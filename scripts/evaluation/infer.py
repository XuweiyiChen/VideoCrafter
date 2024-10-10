import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything
import ptp_utils
from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config
import abc
from pathlib import Path


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, context, video_length, place_in_unet: str):
        # b, c, h, w 
        video_length = 16
        context = rearrange(context, "(b f) d c -> b f d c", f=video_length)
        batch_size = context.shape[0]
        batch_size = batch_size // 2

        if batch_size == 2:
            # Do classifier-free guidance
            hidden_states_uncondition, hidden_states_condition = context.chunk(2)

            if self.cur_step <= self.motion_control_step:
                hidden_states_motion_uncondition = hidden_states_uncondition[
                    1
                ].unsqueeze(0)
            else:
                hidden_states_motion_uncondition = hidden_states_uncondition[
                    0
                ].unsqueeze(0)

            hidden_states_out_uncondition = torch.cat(
                [
                    hidden_states_motion_uncondition,
                    hidden_states_uncondition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Query
            hidden_states_sac_in_uncondition = self.forward(
                hidden_states_uncondition[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out_uncondition = torch.cat(
                [
                    hidden_states_sac_in_uncondition,
                    hidden_states_uncondition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Key & Value

            if self.cur_step <= self.motion_control_step:
                hidden_states_motion_condition = hidden_states_condition[1].unsqueeze(0)
            else:
                hidden_states_motion_condition = hidden_states_condition[0].unsqueeze(0)

            hidden_states_out_condition = torch.cat(
                [
                    hidden_states_motion_condition,
                    hidden_states_condition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Query
            hidden_states_sac_in_condition = self.forward(
                hidden_states_condition[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out_condition = torch.cat(
                [
                    hidden_states_sac_in_condition,
                    hidden_states_condition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Key & Value

            hidden_states_out = torch.cat(
                [hidden_states_out_uncondition, hidden_states_out_condition], dim=0
            )
            hidden_states_sac_out = torch.cat(
                [hidden_states_sac_out_uncondition, hidden_states_sac_out_condition],
                dim=0,
            )

        elif batch_size == 1:
            if self.cur_step <= self.motion_control_step:
                hidden_states_motion = context[1].unsqueeze(0)
            else:
                hidden_states_motion = context[0].unsqueeze(0)

            hidden_states_out = torch.cat(
                [hidden_states_motion, context[1].unsqueeze(0)], dim=0
            )  # Query
            hidden_states_sac_in = self.forward(
                context[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out = torch.cat(
                [hidden_states_sac_in, context[1].unsqueeze(0)], dim=0
            )  # Key & Value

        else:
            raise ValueError("Batch size must be 1 or 2")
        
        context = rearrange(context, "b f d c -> (b f) d c", f=video_length)
        hidden_states_out = rearrange(
            hidden_states_out, "b f d c -> (b f) d c", f=video_length
        )
        hidden_states_sac_out = rearrange(
            hidden_states_sac_out, "b f d c -> (b f) d c", f=video_length
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return hidden_states_out, hidden_states_sac_out, hidden_states_sac_out

    def __init__(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1
        self.motion_control_step = 0


class EmptyControl(AttentionControl):
    def forward(self, context, video_length, place_in_unet):
        return context


class FreeSAC(AttentionControl):
    def forward(self, context, video_length, place_in_unet):
        hidden_states_sac = (
            context[:, 0, :, :].unsqueeze(1).repeat(1, video_length, 1, 1)
        )
        return context
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='/home/tianxia/VideoCrafter/checkpoints/base_512_v1/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default='/home/tianxia/VideoCrafter/configs/inference_t2v_512_v1.0.yaml', help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default="/home/tianxia/VideoCrafter/prompts/data.txt", help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default="/home/tianxia/VideoCrafter/results/512_seed=42", help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    parser.add_argument("--motion_ctrl", type=int, default=1, help="motion control steps")
    return parser

def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    
    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    num_samples = len(prompt_list)
    filename_list = [f"{id+1:04d}" for id in range(num_samples)]

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]

    ## conditional input
    if args.mode == "i2v":
        ## each video or frames dir per prompt
        cond_inputs = get_filelist(args.cond_input, ext='[mpj][pn][4gj]')   # '[mpj][pn][4gj]'
        assert len(cond_inputs) == num_samples, f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
        filename_list = [f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)]
        cond_inputs_rank = [cond_inputs[i] for i in indices]

    filename_list_rank = [filename_list[i] for i in indices]

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in range(0, n_rounds):
        print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx*args.bs
        idx_e = min(idx_s+args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        filenames = prompts
        breakpoint()
        if isinstance(prompts, str):
            prompts = [prompts]
        #prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif args.mode == 'i2v':
            #cond_images = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            cond_images = load_image_batch(cond_inputs_rank[idx_s:idx_e], (args.height, args.width))
            cond_images = cond_images.to(model.device)
            img_emb = model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        # inference
        motion_control = args.motion_ctrl
        motion_control_step = motion_control * args.ddim_steps
        attn_controller = FreeSAC()
        attn_controller.motion_control_step = motion_control_step
        ptp_utils.register_attention_control(model, attn_controller)
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
        ## b,samples,c,t,h,w
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    end = time.time()
    duration = end - start
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")
    output_path = Path(args.savedir, f'loop_duration_{args.savedir.split("/")[-1]}.txt')
    with open(output_path, "w") as file:
        file.write(f"Loop started at: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Loop ended at: {end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Total loop duration: {duration}\n")
    

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s" % now)
    parser = get_parser()
    args = parser.parse_args()
    # Ensure consistent behavior across runs
    seed_everything(args.seed)

    # Assuming these are relevant to your distributed setup or other logic
    rank, gpu_num = 0, 1

    # Run the inference with the manually set arguments
    run_inference(args, gpu_num, rank)