import torch
import numpy as np
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import random 
from scipy.ndimage import gaussian_filter1d
import argparse
import datetime
import script_utils
import os
import faiss

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def init_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def MinMaxNorm(x, min=None, max=None):
    if not min:
        min = x.min()

    if not max:
        max = x.max()

    x = (x - min)/(max-min)
    return x

def smooth_scores(scores_arr, sigma=7):
    for sig in range(1, sigma):
        scores_arr = gaussian_filter1d(scores_arr, sigma=sig)
    return scores_arr

def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=7e-5,
        batch_size=16,
        iterations=10000,
        checkpoint_rate=100,
        log_to_wandb=False,
        log_dir="ddpm_logs",
        project_name="ddpm_avenue_text",
        run_name=run_name,
        dataset="shanghai",  # select shanghai, avenue, and ubnormal
        data_dir="features",
        vis_feat_file="visual_features.h5",
        mot_feat_file="motion_features.h5",
        text_feat_file="text_features.h5",
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule_low=1e-4,
        schedule_high=2e-2,
        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())
    defaults.update(script_utils.graph_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


def get_dynamic_timer(train_dataset):
    train_vis_feature = np.concatenate(train_dataset.features_vis, axis=0)
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(train_vis_feature.shape[1])
    index_vis_features = faiss.index_cpu_to_gpu(res, 0, index)
    index_vis_features.add(train_vis_feature.astype(np.float32))

    return index_vis_features
