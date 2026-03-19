import torch
import torch.nn as nn

class GradientMonitor:
    """
    Monitors gradient statistics during training to detect vanishing/exploding gradients
    or gradient alignment collapse.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_stats = {}

    def capture(self):
        stats = {}
        all_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach()
                stats[f"grad_norm/{name}"] = g.norm().item()
                all_grads.append(g.view(-1))
        
        if all_grads:
            flat_grads = torch.cat(all_grads)
            stats["grad_norm_total"] = flat_grads.norm().item()
            stats["grad_mean"] = flat_grads.mean().item()
            stats["grad_var"] = flat_grads.var().item()
        
        return stats
