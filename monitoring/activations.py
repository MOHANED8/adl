import torch
import torch.nn as nn

class ActivationMonitor:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def capture_stats(self):
        stats = {}
        for name, act in self.activations.items():
            stats[f"act_var/{name}"] = act.var().item()
            stats[f"act_mean/{name}"] = act.mean().item()
            # Fraction of "dead" neurons (zero activations in ReLU context)
            stats[f"dead_frac/{name}"] = (act == 0).float().mean().item()
        return stats

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
