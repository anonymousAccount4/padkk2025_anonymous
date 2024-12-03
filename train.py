import os
import os.path as osp
import ssl
from copy import deepcopy

import numpy as np
import script_utils
import torch
import wandb
from dataset import (
    SHT_Feature_Track_Dataset,
    Ubnormal_Feature_Track_Dataset,
)
from eval import eval
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import create_argparser, init_seed, get_dynamic_timer
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    init_seed()
    args = create_argparser().parse_args()

    device = args.device

    try:
        data_dir = osp.join(args.data_dir, args.dataset)
        if args.dataset == "shanghai":
            train_dataset = SHT_Feature_Track_Dataset(
                args.vis_feat_file,
                args.mot_feat_file,
                args.text_feat_file,
                data_dir,
                osp.join(data_dir, "./SH_Train_OCC.txt"),
            )
            test_dataset = SHT_Feature_Track_Dataset(
                args.vis_feat_file,
                args.mot_feat_file,
                args.text_feat_file,
                data_dir,
                osp.join(data_dir, "./SH_Test_OCC.txt"),
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
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.learning_rate)
        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError(
                    "args.log_to_wandb set to True but args.project_name is None"
                )

            run = wandb.init(
                project=args.project_name,
                entity="treaptofun",
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size
        )
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

        vis_channel = args.vis_channel
        mot_channel = args.mot_channel
        text_channel = args.text_channel

        batch_len = len(train_loader)
        best_auc, best_iter = 0.0, 0
        pbar = tqdm(range(1, args.iterations + 1))
        diffusion.train()
        for iteration in pbar:
            acc_train_loss = 0
            for feat_v, feat_m, feat_t in train_loader:
                feat_v = feat_v.reshape(-1, 8, vis_channel).to(device)
                feat_m = feat_m.reshape(-1, 8, mot_channel).to(device)
                feat_t = feat_t.reshape(-1, 8, text_channel).to(device)

                loss = diffusion(feat_v, feat_m, feat_t)
                acc_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.set_postfix(loss="%.3f" % (acc_train_loss / batch_len))
            if iteration % args.checkpoint_rate == 0:
                diffusion.eval()
                with torch.no_grad():
                    auc = eval(diffusion, test_loader, dynamic_timer, t=args.pred_time, device=device)
                    print(f"cur_epoch:{iteration}, auc:{auc}")
                diffusion.train()

                if auc > best_auc:
                    best_auc = auc
                    best_iter = iteration
                    best_model = deepcopy(diffusion)

        os.makedirs(args.log_dir, exist_ok=True)
        model_filename = f"{args.log_dir}/{args.project_name}-{'%.3f'%best_auc}-iteration-{best_iter}-model.pth"
        torch.save(best_model.state_dict(), model_filename)

        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


if __name__ == "__main__":
    main()
