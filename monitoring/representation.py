import torch
import torch.nn.functional as F
import torch.nn as nn

class RepresentationMonitor:
    """
    Computes representation similarity (CKA) between the current state 
    and a reference (often initial or early) state to detect feature drift.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.reference_activations = None
        self.hooks = []
        self._register_hooks()
        self.current_activations = {}

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.current_activations[name] = output.detach()
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def set_reference(self):
        """Sets the current state as the reference for drift calculation (usually epoch 0)."""
        self.reference_activations = {k: v.clone() for k, v in self.current_activations.items()}

    def capture_drift(self):
        """Computes drift from reference using cosine similarity or CKA proxy."""
        if self.reference_activations is None:
            return {}
        
        stats = {}
        for name, ref in self.reference_activations.items():
            curr = self.current_activations[name]
            # Ensure batch sizes match by taking the minimum
            min_bs = min(ref.size(0), curr.size(0))
            ref_slice = ref[:min_bs].view(min_bs, -1)
            curr_slice = curr[:min_bs].view(min_bs, -1)
            # Simple cosine similarity proxy for drift
            cos_sim = F.cosine_similarity(ref_slice, curr_slice, dim=1).mean().item()
            stats[f"drift/{name}"] = 1.0 - cos_sim
        
        stats["mean_drift"] = sum(stats.values()) / len(stats) if stats else 0.0
        return stats

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
