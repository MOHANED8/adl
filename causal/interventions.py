import torch
import torch.nn as nn

def apply_intervention(model: nn.Module, intervention_type: str, value: float):
    """
    Applies a structural intervention to the model during training.
    
    Args:
        model (nn.Module): The model to intervene upon.
        intervention_type (str): Type of intervention (e.g., 'dropout', 'clipping').
        value (float): The magnitude of the intervention.
    """
    if intervention_type == "dropout":
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = value
    elif intervention_type == "clipping":
        # Note: Gradient clipping is usually handled in the training loop.
        # Here we could set a model property that the trainer checks.
        model.register_buffer("_grad_clip_val", torch.tensor(value))
    elif intervention_type == "weight_decay":
        # Structural intervention on regularization
        pass 
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
