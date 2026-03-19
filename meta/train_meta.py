import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from meta.dataset import MetaDataset
from meta.models import LSTMPredictor, TransformerPredictor
import os
from typing import List

def train_meta(run_dirs: List[str], model_type: str = "lstm", seq_len: int = 8, horizon: int = 3, epochs: int = 10):
    """
    Trains the meta-learning model on gathered trajectories.

    Args:
        run_dirs (List[str]): Directories containing 'metrics.json' files.
        model_type (str): Type of predictor model to use ("lstm" or "transformer").
        seq_len (int): Temporal window size for input sequences.
        horizon (int): Prediction target distance.
        epochs (int): Number of training epochs.
    """
    dataset = MetaDataset(run_dirs, seq_len=seq_len, horizon=horizon)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = dataset[0][0].shape[1]
    if model_type == "lstm":
        model = LSTMPredictor(input_dim)
    else:
        model = TransformerPredictor(input_dim)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Meta Epoch {epoch}: Loss={total_loss/len(loader):.6f}")

    os.makedirs("meta_models", exist_ok=True)
    torch.save(model.state_dict(), f"meta_models/{model_type}_predictor.pt")
    print(f"Meta model {model_type} saved.")

if __name__ == "__main__":
    # Example usage (assuming results/ exist)
    run_dirs = [os.path.join("results", d) for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
    if run_dirs:
        train_meta(run_dirs)
