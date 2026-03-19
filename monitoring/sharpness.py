import torch

def hutchinson_fisher_trace(model, images, labels, criterion, num_samples=1):
    """Estimates the trace of the Fisher Information Matrix using Hutchinson's estimator."""
    trace = 0.0
    model.eval() # Ensure no dropout/batchnorm noise
    
    for _ in range(num_samples):
        v = [torch.randn_like(p) for p in model.parameters() if p.requires_grad]
        
        # We use the finite difference approximation of HVP: H v = (grad(x+hv) - grad(x-hv)) / 2h
        # But for Fisher, it's grad @ grad.T. Trace(Fisher) = E[v.T grad grad.T v] = E[(grad.T v)^2]
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=False)
        
        gv = sum((g * vi).sum() for g, vi in zip(grads, v))
        trace += (gv**2).item()
        
    model.train()
    return trace / num_samples

class SharpnessMonitor:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def capture(self, images, labels):
        stats = {}
        stats["fisher_trace"] = hutchinson_fisher_trace(self.model, images, labels, self.criterion)
        return stats
