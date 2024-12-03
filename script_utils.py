import argparse

import torch
import torch.nn.functional as F
import torchvision
from diffusion import (
    MDVAD,
    GaussianDiffusion,
    generate_cosine_schedule,
    generate_linear_schedule,
)
from model import UnetModel
from torchvision import transforms


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        vis_channel=1024,
        mot_channel=512,
        text_channel=768,
        time_emb_dim=512,
        pred_time=40,  # shanghai: 40, ubnormal: 40, avenue: 250
        two_cond=True,  # complementary use of MD and TD
        cond_type="motion",  # if two_cond is False, choose MD(motion) or TD(text).
        mode="test",  # select test or train mode
        mean_type="xstart",
        channel_mults=[2, 2, 2],
        num_res_blocks=1,
        # select pretraiend weight of MD and TD.
        # shanghai(0.857)
        motion_pretrained="./pretrained/shanghai-auc-0.8435-MD-model.pth",
        text_pretrained="./pretrained/shanghai-auc-0.8422-TD-model.pth",
        # avenue(0.9037)
        # motion_pretrained="./pretrained/avenue-auc-0.894-MD-model.pth",
        # text_pretrained="./pretrained/avenue-auc-0.896-TD-model.pth",
        # pretrained="./pretrained/ddpm_avenue-0.870-iteration-3600-model.pth",
        # ubnormal(0.6536)
        # motion_pretrained="./pretrained/ubnormal-auc-0.6267-MD-model.pth",
        # text_pretrained="./pretrained/ubnormal-auc-0.6243-TD-model.pth",
    )
    return defaults

def graph_defaults():
    defaults = dict(
        layer_channels=[64, 128, 128, 256, 256, 512, 512],
        num_coords=2,
        n_frames=8,
        n_joints=17,
        strategy="distance",
        max_hops=1,
    )
    return defaults

def get_diffusion_from_args(args):
    if args.cond_type == "text":
        cond_dim = args.text_channel
        time_ebmd_dim = cond_dim
    elif args.cond_type == "motion":
        cond_dim = args.mot_channel
        time_ebmd_dim = cond_dim
    elif args.cond_type == "none":
        cond_dim = None
        time_ebmd_dim = args.time_emb_dim
    else:
        NotImplementedError(args.cond_type)

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    if args.two_cond:
        diffusion = MDVAD(
            vis_channel=args.vis_channel,
            mot_channel=args.mot_channel,
            text_channel=args.text_channel,
            betas=betas,
            ch_mult=args.channel_mults,
            num_res_blocks=args.num_res_blocks,
            motion_pretrained=args.motion_pretrained,
            text_pretrained=args.text_pretrained,
            mean_type=args.mean_type,
            loss_type=args.loss_type,
            mode=args.mode,
        )
        # diffusion.load_state_dict(torch.load(args.pretrained))
    else:
        model = UnetModel(
            in_channel=args.vis_channel,
            time_emb_dim=time_ebmd_dim,
            cond_dim=cond_dim,
            ch_mult=args.channel_mults,
            num_res_blocks=args.num_res_blocks,
        )

        diffusion = GaussianDiffusion(
            model,
            args.vis_channel,
            betas,
            mean_type=args.mean_type,
            loss_type=args.loss_type,
            cond=args.cond_type,
        )

        if args.mode == "test" and args.cond_type == "text":
            diffusion.load_state_dict(torch.load(args.text_pretrained))

        elif args.mode == "test" and args.cond_type == "motion":
            diffusion.load_state_dict(torch.load(args.motion_pretrained))
    return diffusion
