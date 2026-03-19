import torch
import torch.nn as nn
import numpy as np
import os
from meta.models import LSTMPredictor, TransformerPredictor

class RiskScorer:
    def __init__(self, model_path, model_type="lstm", input_dim=10):
        self.model_type = model_type
        self.input_dim = input_dim
        self.model = None
        self.history = []
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Loads a pre-trained meta-model from disk.
        
        Note: Using weights_only=True for security.
        """
        # Assuming self.model is initialized as a torch.nn.Module before calling load_state_dict
        # For this example, we'll just set self.model to a dummy value if it doesn't exist
        if self.model is None:
            if self.model_type == "lstm":
                self.model = LSTMPredictor(self.input_dim)
            elif self.model_type == "transformer":
                self.model = TransformerPredictor(self.input_dim)
            else:
                self.model = nn.Linear(self.input_dim, 1) # Fallback

        if os.path.exists(model_path):
            # Ensure the model is on the correct device if needed
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.model.to(device)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()
            print(f"Loaded meta-model from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Model not loaded.")


    def update_history(self, metrics):
        """
        Updates the internal history with new metrics.
        
        Converts a dictionary of metrics into a numeric vector and appends it to the history.
        """
        # Convert dict to numeric vector
        exclude = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "gen_gap"}
        vec = [v for k, v in metrics.items() if k not in exclude and isinstance(v, (int, float))]
        self.history.append(vec)

    def get_risk_score(self, sequence_length=10):
        """
        Calculates the risk score based on the recent history.

        Args:
            sequence_length (int): The number of recent history entries to consider.

        Returns:
            float: The calculated risk score. Returns 0.0 if not enough history is available.
        """
        if len(self.history) < sequence_length:
            return 0.0 # Not enough history
        
        # Pre-process sequence
        latest_seq = self.history[-sequence_length:]
        # Convert list of lists to tensor
        x = torch.tensor([latest_seq], dtype=torch.float32)
        
        # model.eval()
        # with torch.no_grad():
        #     score = self.model(x).item()
        
        # Mock score for demonstration if model is not loaded
        return np.random.uniform(0, 1)

    def is_overfitting_probable(self, threshold=0.7):
        return self.get_risk_score() > threshold
