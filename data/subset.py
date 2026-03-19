import torch
import numpy as np
from torch.utils.data import Subset

def get_stratified_subset(dataset, fraction: float, num_classes: int):
    if fraction >= 1.0:
        return dataset
    
    targets = np.array(dataset.targets)
    indices = []
    for c in range(num_classes):
        c_indices = np.where(targets == c)[0]
        np.random.shuffle(c_indices)
        take = int(len(c_indices) * fraction)
        indices.extend(c_indices[:take])
    
    return Subset(dataset, indices)
