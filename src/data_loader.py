import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import color

class CustomDataset(Dataset):
    def __init__(self, preloaded_data, num_bins=313, points_per_scan=30000, points_per_batch=2048, scans_per_batch=16):
        self.num_bins = num_bins
        self.points_per_scan = points_per_scan
        self.points_per_batch = points_per_batch
        self.scans_per_batch = scans_per_batch
        self.points_per_scan_in_batch = self.points_per_batch // self.scans_per_batch

        data = preloaded_data

        self.locations = torch.tensor(data['sampled_points'], dtype=torch.float32)
        self.gt_sdf = torch.tensor(data['gt_sdfs'], dtype=torch.float32)
        self.gt_confidence = torch.tensor(data['gt_confidences'], dtype=torch.float32)
        self.gt_semantics = torch.tensor(data['gt_semantics'], dtype=torch.float32)
        self.gt_color = torch.tensor(self.rgb_to_lab_discretized(data['gt_colors']), dtype=torch.long)

        self.scan_indices = torch.tensor(data['scan_indices'], dtype=torch.long)
        self.point_clouds = data['observed_pcd']
        self.mel_spectrograms = data['mel_spectrum']
        self.gt_trav = data['gt_trav']

        self.scans = self.group_by_scan()

        # Initialize to store all points' indices from each scan
        self.all_points_indices = []

        # Shuffle points at the beginning of each epoch
        self._shuffle_points()

    def group_by_scan(self):
        scans = {}
        for i, scan_idx in enumerate(self.scan_indices):
            scan_idx = scan_idx.item()
            if scan_idx not in scans:
                scans[scan_idx] = []
            scans[scan_idx].append(i)
        return scans

    def _shuffle_points(self):
        self.all_points_indices = []
        for scan_idx, point_indices in self.scans.items():
            np.random.shuffle(point_indices)  # Shuffle the points within each scan
            self.all_points_indices.append(point_indices)

    def normalize_sdf(self, sdf_tensor, min_val=-6, max_val=6):
        clipped_sdf = torch.clamp(sdf_tensor, min_val, max_val)
        min_max_range = max_val - min_val
        normalized_sdf = 2 * ((clipped_sdf - min_val) / min_max_range) - 1
        return normalized_sdf

    def rgb_to_lab_discretized(self, rgb_colors):
        colorless_mask = (rgb_colors == [-1, -1, -1]).all(axis=-1)
        lab_colors = color.rgb2lab(rgb_colors)  # Convert normalized RGB to LAB

        lab_colors_normalized = (lab_colors + [0, 128, 128]) / [100, 255, 255]
        lab_colors_normalized = np.clip(lab_colors_normalized, 0, 1)

        lab_colors_discretized = (lab_colors_normalized * (self.num_bins - 1)).astype(int)
        lab_colors_discretized[colorless_mask] = self.num_bins - 1

        assert np.all(lab_colors_discretized >= 0) and np.all(lab_colors_discretized < self.num_bins), "Discretized LAB values out of range"
        return lab_colors_discretized

    def __len__(self):
        return len(self.locations) // self.points_per_batch

    def __getitem__(self, idx):
        num_available_scans = len(self.all_points_indices)
        num_scans_to_sample = min(self.scans_per_batch, num_available_scans)
        
        selected_scans = np.random.choice(num_available_scans, num_scans_to_sample, replace=False)

        selected_points = []
        for scan_idx in selected_scans:
            points = self.all_points_indices[scan_idx][:self.points_per_scan_in_batch]
            self.all_points_indices[scan_idx] = self.all_points_indices[scan_idx][self.points_per_scan_in_batch:]
            selected_points.extend(points)

        selected_points = torch.tensor(selected_points, dtype=torch.long)  # Ensure long dtype for indexing

        locations = self.locations[selected_points]
        gt_sdf = self.gt_sdf[selected_points]
        gt_confidence = self.gt_confidence[selected_points]
        gt_semantics = self.gt_semantics[selected_points]
        gt_color = self.gt_color[selected_points]

        point_clouds = [torch.tensor(self.point_clouds[scan_idx], dtype=torch.float32) for scan_idx in selected_scans]
        mel_spectrograms = [torch.tensor(self.mel_spectrograms[scan_idx], dtype=torch.float32).permute(2, 0, 1) for scan_idx in selected_scans]
        gt_travs = [torch.tensor(self.gt_trav[scan_idx], dtype=torch.float32) for scan_idx in selected_scans]

        point_clouds = torch.stack(point_clouds)
        mel_spectrograms = torch.stack(mel_spectrograms)
        gt_travs = torch.stack(gt_travs)

        return {
            'locations': locations,
            'point_clouds': point_clouds,
            'gt_sdf': gt_sdf,
            'gt_confidence': gt_confidence,
            'gt_semantics': gt_semantics,
            'gt_color': gt_color,
            'mel_spectrograms': mel_spectrograms,
            'gt_travs': gt_travs
        }

    def on_epoch_start(self):
        self._shuffle_points()

def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {key: [] for key in keys}

    for item in batch:
        for key in keys:
            collated[key].append(item[key])

    # Stack tensors
    for key in ['point_clouds', 'mel_spectrograms', 'gt_travs']:
        collated[key] = torch.stack(collated[key])

    # Concatenate tensors
    for key in ['locations', 'gt_sdf', 'gt_confidence', 'gt_semantics', 'gt_color']:
        collated[key] = torch.cat(collated[key], dim=0)

    return collated
