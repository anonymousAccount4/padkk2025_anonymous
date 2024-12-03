import math
import os
import os.path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import script_utils
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import (
    SHT_Feature_Track_Dataset,
    Ubnormal_Feature_Track_Dataset,
)
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    MinMaxNorm,
    create_argparser,
    get_dynamic_timer,
    init_seed,
    smooth_scores,
)


def plot_result(gt, pred, key, score=None, save_dir="./results/shanghai/text"):
    if key == '01_0063':
        print(pred.max(), pred.min())
    x = np.arange(len(gt))
    plt.figure(figsize=(10, 5))
    plt.ylim([0, 1])
    anomaly_idx = np.where(gt == 1)[0]
    plt.bar(anomaly_idx, 1, width=1, color="r", alpha=0.5, label="Ground-truth")
    plt.plot(x, pred, "b", label=f"MDVAD: AUC={score:.3f}")
    plt.xlabel("Frame number", fontsize=16)
    plt.ylabel("Anomaly Score", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f"{save_dir}/{key}.png")
    plt.close()

def preprocessing(x):
    org_l = len(x)
    new_l = math.ceil(org_l / 8) * 8
    x = (
        F.pad(x.unsqueeze(0), (0, 0, 0, new_l - org_l), "replicate")
        .squeeze(0)
        .reshape(new_l // 8, 8, -1)
    )
    return x, org_l

def eval(
    model,
    test_loader,
    dynamic_timer,
    t=50,
    hr_mask=None,
    video_split=None,
    save_dir=None,
    device="cpu",
):
    gt_all, pred_all = [], []
    for feat_v, feat_m, feat_t, label, chunk, info, video in test_loader:
        label = label.squeeze(0)
        feat_v = feat_v.to(device).reshape(-1, feat_v.shape[-1])
        feat_m = feat_m.to(device).reshape(-1, feat_m.shape[-1])
        feat_t = feat_t.to(device).reshape(-1, feat_t.shape[-1])
        feat_v_pad, len_v = preprocessing(feat_v)
        feat_m_pad, _ = preprocessing(feat_m)
        feat_t_pad, _ = preprocessing(feat_t)

        output = model.sample(feat_v_pad, feat_m_pad, feat_t_pad, dynamic_timer, t)
        output = output[:len_v]
        output = torch.mean((feat_v - output) ** 2, 1)
        tracks = torch.split(output, chunk)
        max_pred = torch.zeros(len(label))
        for i in range(len(tracks)):
            pred = F.interpolate(
                tracks[i].reshape(1, 1, -1), scale_factor=16, mode="nearest"
            ).reshape(-1)
            start, end = info[i]
            pred_ = torch.zeros(len(label))
            if end - start + 1 > len(pred):
                pred = F.pad(
                    pred.reshape(1, 1, -1),
                    (0, (end - start + 1) - len(pred)),
                    "replicate",
                ).reshape(-1)
            pred_[start : end + 1] = pred
            max_pred = torch.max(torch.stack([max_pred, pred_]), dim=0)[0]
        gt_all.append(label)
        max_pred = max_pred.detach().cpu().numpy()
        max_pred = smooth_scores(max_pred)
        pred_all.append(max_pred)
    pred_all, gt_all = np.concatenate(pred_all, axis=0), np.concatenate(gt_all, axis=0)

    if hr_mask is not None:
        pred_all = pred_all[hr_mask]
        gt_all = gt_all[hr_mask]
        
    pred_all = MinMaxNorm(pred_all)
    fpr, tpr, _ = roc_curve(gt_all, pred_all, pos_label=1)
    total_score = auc(fpr, tpr)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for key, (sf, ef) in video_split.items():
            gt = gt_all[sf:ef]
            pred = pred_all[sf:ef]

            if gt.sum() == len(gt) or gt.sum() == 0:
                score = 0.0

            else:
                score = roc_auc_score(gt, pred)

            plot_result(gt, pred, key, score, save_dir)

    return total_score


def main():
    args = create_argparser().parse_args()
    device = args.device

    # save_dir = f"./results/{args.dataset}/text_motion"
    save_dir = None
    data_dir = osp.join(args.data_dir, args.dataset)
    if args.dataset == "shanghai":
        train_dataset = SHT_Feature_Track_Dataset(
            args.vis_feat_file,
            args.mot_feat_file,
            args.text_feat_file,
            data_dir,
            "./SH_Train_OCC.txt",
        )
        test_dataset = SHT_Feature_Track_Dataset(
            args.vis_feat_file,
            args.mot_feat_file,
            args.text_feat_file,
            data_dir,
            "./SH_Test_OCC.txt",
        )

    elif args.dataset == "ubnormal":
        train_dataset = Ubnormal_Feature_Track_Dataset(
            args.vis_feat_file,
            args.mot_feat_file,
            args.text_feat_file,
            data_dir,
            "train",
        )
        test_dataset = Ubnormal_Feature_Track_Dataset(
            args.vis_feat_file,
            args.mot_feat_file,
            args.text_feat_file,
            data_dir,
            "test",
        )

    dynamic_timer = get_dynamic_timer(train_dataset)

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    diffusion.eval()
    hr_mask = np.load(f"./features/{args.dataset}/hr_mask.npy")
    
    i = 69
    init_seed(i)
    with torch.no_grad():
        auc = eval(
            diffusion,
            test_loader,
            dynamic_timer,
            t=50,
            hr_mask=None,
            video_split=test_dataset.video_split,
            save_dir=save_dir,
            device=device,
        )
        print(f"{i} : {auc}")

if __name__ == "__main__":
    main()
