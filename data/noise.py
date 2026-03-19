import torch
import numpy as np
import random
from config import NoiseType

def inject_label_noise(dataset, noise_type: NoiseType, level: float, num_classes: int):
    if noise_type == NoiseType.NONE or level <= 0:
        return dataset
    
    noisy_targets = list(dataset.targets)
    n = len(noisy_targets)
    num_to_flip = int(n * level)
    indices_to_flip = random.sample(range(n), num_to_flip)
    
    if noise_type == NoiseType.SYMMETRIC:
        for idx in indices_to_flip:
            current_label = noisy_targets[idx]
            new_label = random.choice([l for l in range(num_classes) if l != current_label])
            noisy_targets[idx] = new_label
    elif noise_type == NoiseType.ASYMMETRIC:
        # Simple asymmetric rule for CIFAR-10 (standard in literature)
        # truck(9) -> automobile(1), bird(2) -> airplane(0), cat(3) -> dog(5), dog(5) -> cat(3), deer(4) -> horse(7)
        mapping = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
        for idx in indices_to_flip:
            curr = noisy_targets[idx]
            if curr in mapping:
                noisy_targets[idx] = mapping[curr]
    
    dataset.targets = noisy_targets
    return dataset
