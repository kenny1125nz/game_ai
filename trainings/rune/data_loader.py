import os
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from torchvision import datasets

def joined_load(synthetic_path: str, error_path: str, transform):
    syn_ds = datasets.ImageFolder(root=synthetic_path, transform=transform)
    err_ds = datasets.ImageFolder(root=error_path, transform=transform)

    # 4. MERGE (The standard PyTorch way)
    combined = ConcatDataset([syn_ds, err_ds])

    # 6. WEIGHTING
    len_syn = len(syn_ds)
    len_err = len(err_ds)
    
    # Give real-world 'error' data 5x more priority
    weights = ([1.0] * len_syn) + ([3.0] * len_err)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len_syn+len_err,
        replacement=True
    )

    print(f"Merged: {len_syn} Synthetic + {len_err} Error = {len(combined)} Total.")
    return combined, sampler, syn_ds.classes,syn_ds.class_to_idx