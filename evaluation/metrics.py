import torch
import numpy as np
import pandas as pd
import json
import os

def compare_early_stopping(run_id, oracle_epoch):
    """Compares the prediction timestamp of the diagnostic system with the Oracle onset."""
    # Oracle onset = epoch with minimum val_loss
    # diagnostic_onset = epoch where risk score first crossed threshold
    pass

def calculate_generalization_gain(base_val_err, diagnostic_val_err):
    return base_val_err - diagnostic_val_err
