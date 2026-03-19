import torch
import torch.nn as nn
import os
import numpy as np

class SnapshotMonitor:
    """
    Captures full layer-wise activations for a fixed subset of validation data.
    Used for offline CKA and representation drift analysis.
    
    Args:
        model (nn.Module): The model to monitor.
        sample_subset (torch.Tensor): Fixed batch of validation images.
    """
    def __init__(self, model: nn.Module, sample_subset: torch.Tensor):
        self.model = model
        self.sample_subset = sample_subset  # Fixed batch of validation images
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def capture_and_save(self, run_dir, epoch):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(self.sample_subset)
        
        snapshot_dir = os.path.join(run_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        np.savez(os.path.join(snapshot_dir, f"epoch_{epoch}.npz"), **self.activations)
        self.model.train()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
