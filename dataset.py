import os
import os.path as osp
import pickle
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class SHT_Feature_Track_Dataset(Dataset):
    def __init__(
        self, vis_feat_file, mot_feat_file, text_feat_file, data_dir, split_txt
    ):
        super(SHT_Feature_Track_Dataset, self).__init__()
        self.h5_path_vis = osp.join(data_dir, vis_feat_file)
        self.h5_path_mot = osp.join(data_dir, mot_feat_file)
        self.h5_path_text = osp.join(data_dir, text_feat_file)
        self.videos = open(split_txt, "r").read().split("\n")
        self.split = split_txt.split("_")[1].split(".")[0]
        self.clip_length = 8
        self.features_vis = []
        self.features_text = []
        self.features_mot = []
        self.all_seqs = []
        self.labels = []
        self.video_split = {}
        self.chunks = {}
        self.data_dir = data_dir
        self.tracks_info = pickle.load(
            open(osp.join(self.data_dir, "track_info.pickle"), "rb")
        )
        self.infos = {}
        self.load_feat()

    def load_feat(self):
        if self.split == "Train":
            for video in self.videos:
                tracks = list(h5py.File(self.h5_path_vis, "r")[video].keys())
                feats_vis, feats_mot, feats_text = [], [], []
                for track in tracks:
                    feat_vis = np.array(
                        h5py.File(self.h5_path_vis, "r")[f"{video}/{track}"]
                    )
                    feats_vis.append(feat_vis)
                    feat_mot = np.array(
                        h5py.File(self.h5_path_mot, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_mot.append(feat_mot)
                    feat_text = np.array(
                        h5py.File(self.h5_path_text, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_text.append(feat_text)
                feats_mot = np.concatenate(feats_mot, axis=0)
                feats_vis = np.concatenate(feats_vis, axis=0)
                feats_text = np.concatenate(feats_text, axis=0)
                random_seq = list(range(len(feats_vis) - self.clip_length + 1))
                random.shuffle(random_seq)
                self.all_seqs.append(random_seq)
                self.features_vis.append(feats_vis)
                self.features_mot.append(feats_mot)
                self.features_text.append(feats_text)

        else:
            sum_frame = 0
            for video in self.videos:
                tracks = list(h5py.File(self.h5_path_vis, "r")[video].keys())
                # tracks = ["2779.npy"]
                feats_vis, feats_mot, feats_text, noise = [], [], [], []
                for track in tracks:
                    feat_vis = np.array(
                        h5py.File(self.h5_path_vis, "r")[f"{video}/{track}"]
                    )
                    feats_vis.append(feat_vis)
                    feat_mot = np.array(
                        h5py.File(self.h5_path_mot, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_mot.append(feat_mot)
                    feat_text = np.array(
                        h5py.File(self.h5_path_text, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_text.append(feat_text)
                    start, end = self.tracks_info[f"{video}/{track[:-4]}"].values()

                    if f"{video}" not in self.chunks:
                        self.chunks[f"{video}"] = [len(feat_vis)]
                        self.infos[f"{video}"] = [(start, end)]
                    else:
                        self.chunks[f"{video}"].append(len(feat_vis))
                        self.infos[f"{video}"].append((start, end))

                self.features_vis.append(np.concatenate(feats_vis, axis=0))
                self.features_mot.append(np.concatenate(feats_mot, axis=0))
                self.features_text.append(np.concatenate(feats_text, axis=0))

                label = np.load(osp.join(self.data_dir, f"labels/{video}.npy"))
                self.labels.append(label)
                self.video_split[video] = [sum_frame, sum_frame + label.shape[0]]
                sum_frame += label.shape[0]

    def __getitem__(self, idx):
        feat_vis = torch.from_numpy(self.features_vis[idx]).float()
        feat_mot = torch.from_numpy(self.features_mot[idx]).float()
        feat_text = torch.from_numpy(self.features_text[idx]).float()

        if self.split == "Test":
            video = self.videos[idx]
            chunk, info = self.chunks[video], self.infos[video]
            label = self.labels[idx]
            return feat_vis, feat_mot, feat_text, label, chunk, info, video

        start = self.all_seqs[idx].pop()
        end = start + self.clip_length
        if len(self.all_seqs[idx]) == 0:
            self.all_seqs[idx] = list(range(len(feat_vis) - self.clip_length + 1))
            random.shuffle(self.all_seqs[idx])

        return feat_vis[start:end], feat_mot[start:end], feat_text[start:end]

    def __len__(self):
        return len(self.features_vis)


class Avenue_Feature_Track_Dataset(Dataset):
    def __init__(
        self, vis_feat_file, mot_feat_file, text_feat_file, data_dir, mode="train", vid_res=[640, 360], symm_range=True
    ):
        super().__init__()
        self.h5_path_vis = osp.join(data_dir, mode, vis_feat_file)
        self.h5_path_mot = osp.join(data_dir, mode, mot_feat_file)
        self.h5_path_text = osp.join(data_dir, mode, text_feat_file)
        self.videos = list(h5py.File(self.h5_path_vis, "r").keys())
        self.split = mode
        self.clip_length = 8
        self.vid_res = vid_res
        self.symm_range = symm_range
        self.features_vis = []
        self.features_text = []
        self.features_mot = []
        self.all_seqs = []
        self.labels = []
        self.video_split = {}
        self.chunks = {}
        self.data_dir = osp.join(data_dir, mode)
        if self.split == "test":
            self.infos = pickle.load(open(f"{self.data_dir}/test_info.pkl", "rb"))
        self.load_feat()

    def normalize_pose(self, pose_data):
        """
        Normalize keypoint values to the range of [-1, 1]
        :param pose_data: Formatted as [T, V, F], e.g. (Frames=12, 18, 3)
        :param vid_res:
        :param symm_range:
        :return:
        """
        
        vid_res_wconf = self.vid_res
        norm_factor = np.array(vid_res_wconf)
        pose_data_normalized = pose_data / norm_factor
        pose_data_centered = pose_data_normalized
        if self.symm_range:  # Means shift data to [-1, 1] range
            pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

        pose_data_zero_mean = pose_data_centered
        # return pose_data_zero_mean
        pose_data_mean, pose_data_std = pose_data_centered[..., :2].mean(axis=(0, 1)), pose_data_centered[..., 1].std(axis=(0, 1))
        pose_data_zero_mean[..., :2] = (pose_data_centered[..., :2] - pose_data_mean[None, None, :]) / pose_data_std[None, None, None]
        return pose_data_zero_mean

    def load_feat(self):
        if self.split == "train":
            for video in self.videos:
                feats_vis = np.array(
                    h5py.File(self.h5_path_vis, "r")[f"{video}"]
                )
                feats_mot = np.array(
                    h5py.File(self.h5_path_mot, "r")[f"{video}"]
                )
                feats_mot = self.normalize_pose(feats_mot)
                feats_text = np.array(
                    h5py.File(self.h5_path_text, "r")[f"{video}"]
                )
                random_seq = list(range(len(feats_vis) - self.clip_length + 1))
                random.shuffle(random_seq)
                self.all_seqs.append(random_seq)
                self.features_vis.append(feats_vis)
                self.features_mot.append(feats_mot)
                self.features_text.append(feats_text)

        else:
            sum_frame = 0
            for video in self.videos:
                feats_vis = np.array(h5py.File(self.h5_path_vis, "r")[f"{video}"])
                feats_mot = np.array(h5py.File(self.h5_path_mot, "r")[f"{video}"])
                feats_text = np.array(h5py.File(self.h5_path_text, "r")[f"{video}"])
        
                self.features_vis.append(feats_vis)
                self.features_mot.append(feats_mot)
                self.features_text.append(feats_text)

                label = np.load(osp.join(self.data_dir, f"labels/{video}"))
                self.labels.append(label)
                self.video_split[video] = [sum_frame, sum_frame + label.shape[0]]
                sum_frame += label.shape[0]

    def __getitem__(self, idx):
        feat_vis = torch.from_numpy(self.features_vis[idx]).float()
        feat_mot = torch.from_numpy(self.features_mot[idx]).float()
        feat_text = torch.from_numpy(self.features_text[idx]).float()

        if self.split == "test":
            video, label = self.videos[idx], self.labels[idx]
            info = self.infos[video.split(".")[0]]
            return feat_vis, feat_mot, feat_text, label, info, video

        start = self.all_seqs[idx].pop()
        end = start + self.clip_length
        if len(self.all_seqs[idx]) == 0:
            self.all_seqs[idx] = list(range(len(feat_vis) - self.clip_length + 1))
            random.shuffle(self.all_seqs[idx])

        return feat_vis[start:end], feat_mot[start:end].permute(2, 0, 1), feat_text[start:end]

    def __len__(self):
        return len(self.features_vis)


class Ubnormal_Feature_Track_Dataset(Dataset):
    def __init__(
        self, vis_feat_file, mot_feat_file, text_feat_file, data_dir, mode="train"
    ):
        super(Ubnormal_Feature_Track_Dataset, self).__init__()
        self.h5_path_vis = osp.join(data_dir, mode, vis_feat_file)
        self.h5_path_mot = osp.join(data_dir, mode, mot_feat_file)
        self.h5_path_text = osp.join(data_dir, mode, text_feat_file)
        self.videos = list(h5py.File(self.h5_path_vis, "r").keys())
        self.split = mode
        self.clip_length = 8
        self.features_vis = []
        self.features_text = []
        self.features_mot = []
        self.all_seqs = []
        self.labels = []
        self.video_split = {}
        self.chunks = {}
        self.data_dir = osp.join(data_dir, mode)
        self.tracks_info = pickle.load(
            open(osp.join(self.data_dir, "track_info.pickle"), "rb")
        )
        self.infos = {}
        self.frames = pickle.load(open("ubnormal_frames.pickle", "rb"))
        self.load_feat()

    def load_feat(self):
        if self.split == "train":
            for video in self.videos:
                tracks = list(h5py.File(self.h5_path_vis, "r")[video].keys())
                feats_vis, feats_mot, feats_text = [], [], []
                for track in tracks:
                    feat_vis = np.array(
                        h5py.File(self.h5_path_vis, "r")[f"{video}/{track}"]
                    )
                    feats_vis.append(feat_vis)
                    feat_mot = np.array(
                        h5py.File(self.h5_path_mot, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_mot.append(feat_mot)
                    feat_text = np.array(
                        h5py.File(self.h5_path_text, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_text.append(feat_text)
                feats_vis = np.concatenate(feats_vis, axis=0)
                feats_mot = np.concatenate(feats_mot, axis=0)
                feats_text = np.concatenate(feats_text, axis=0)
                if len(feats_vis) < self.clip_length:
                    l = len(feats_vis)
                    feats_vis = np.pad(
                        feats_vis, ((0, self.clip_length - l), (0, 0)), "edge"
                    )
                    feats_mot = np.pad(
                        feats_mot, ((0, self.clip_length - l), (0, 0)), "edge"
                    )
                    feats_text = np.pad(
                        feats_text, ((0, self.clip_length - l), (0, 0)), "edge"
                    )
                random_seq = list(range(len(feats_vis) - self.clip_length + 1))
                random.shuffle(random_seq)
                self.all_seqs.append(random_seq)
                self.features_vis.append(feats_vis)
                self.features_mot.append(feats_mot)
                self.features_text.append(feats_text)

        else:
            sum_frame = 0
            for video in self.videos:
                tracks = list(h5py.File(self.h5_path_vis, "r")[video].keys())
                feats_vis, feats_mot, feats_text = [], [], []
                for track in tracks:
                    feat_vis = np.array(
                        h5py.File(self.h5_path_vis, "r")[f"{video}/{track}"]
                    )
                    feats_vis.append(feat_vis)
                    feat_mot = np.array(
                        h5py.File(self.h5_path_mot, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_mot.append(feat_mot)
                    feat_text = np.array(
                        h5py.File(self.h5_path_text, "r")[f"{video}/{track}"]
                    )[: len(feat_vis)]
                    feats_text.append(feat_text)
                    start, end = self.tracks_info[f"{video}/{track[:-4]}"].values()
                    if f"{video}" not in self.chunks:
                        self.chunks[f"{video}"] = [len(feat_vis)]
                        self.infos[f"{video}"] = [(start, end)]
                    else:
                        self.chunks[f"{video}"].append(len(feat_vis))
                        self.infos[f"{video}"].append((start, end))

                self.features_vis.append(np.concatenate(feats_vis, axis=0))
                self.features_mot.append(np.concatenate(feats_mot, axis=0))
                self.features_text.append(np.concatenate(feats_text, axis=0))

                if video.split("_")[0] == "abnormal":
                    label = np.load(osp.join(self.data_dir, f"labels/{video}.npy"))
                elif video.split("_")[0] == "normal":
                    frame_len = self.frames[video]
                    label = np.zeros((frame_len,))
                self.labels.append(label)
                self.video_split[video] = [sum_frame, sum_frame + label.shape[0]]
                sum_frame += label.shape[0]

    def __getitem__(self, idx):
        feat_vis = torch.from_numpy(self.features_vis[idx]).float()
        feat_mot = torch.from_numpy(self.features_mot[idx]).float()
        feat_text = torch.from_numpy(self.features_text[idx]).float()

        if self.split == "test":
            video = self.videos[idx]
            chunk, info = self.chunks[video], self.infos[video]
            label = self.labels[idx]
            return feat_vis, feat_mot, feat_text, label, chunk, info, video

        start = self.all_seqs[idx].pop()
        end = start + self.clip_length
        if len(self.all_seqs[idx]) == 0:
            self.all_seqs[idx] = list(range(len(feat_vis) - self.clip_length + 1))
            random.shuffle(self.all_seqs[idx])

        return feat_vis[start:end], feat_mot[start:end], feat_text[start:end]

    def __len__(self):
        return len(self.features_vis)
