import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from config import ExperimentConfig
from models.factory import get_model
from data.loaders import get_loaders
from monitoring.gradients import GradientMonitor
from monitoring.activations import ActivationMonitor
from monitoring.representation import RepresentationMonitor
from monitoring.sharpness import SharpnessMonitor
from monitoring.logger import TrajectoryLogger
from analysis.snapshots import SnapshotMonitor

def train(config: ExperimentConfig, run_id: str):
    """
    Orchestrates the training of a single model instance.
    
    Args:
        config (ExperimentConfig): Configuration object containing model, data, and training params.
        run_id (str): Unique identifier for this experimental run.
    """
    # Data
    train_loader, val_loader, num_classes = get_loaders(
        config.dataset, config.data_fraction, config.noise_type, config.noise_level, config.batch_size
    )
    # Fixed subset for snapshots (e.g., first batch of val_loader)
    snapshot_subset, _ = next(iter(val_loader))
    snapshot_subset = snapshot_subset.to(config.device)

    # Model
    model = get_model(config.model_type, num_classes=num_classes, dropout=config.dropout_rate).to(config.device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Monitors
    grad_monitor = GradientMonitor(model)
    act_monitor = ActivationMonitor(model)
    rep_monitor = RepresentationMonitor(model)
    sharp_monitor = SharpnessMonitor(model, criterion)
    snap_monitor = SnapshotMonitor(model, snapshot_subset)
    logger = TrajectoryLogger(os.path.join("results", run_id))

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Set reference representations at start of first epoch
            if epoch == 0 and i == 0:
                rep_monitor.set_reference()

        # End of epoch signals
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": 100. * correct / total,
        }
        
        # Internal signals
        metrics.update(grad_monitor.capture())
        metrics.update(act_monitor.capture_stats())
        metrics.update(rep_monitor.capture_drift())
        # Sharpness estimation on a small subset of the val_loader for speed
        val_images, val_labels = next(iter(val_loader))
        metrics.update(sharp_monitor.capture(val_images.to(config.device), val_labels.to(config.device)))

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        metrics["val_loss"] = val_loss / len(val_loader)
        metrics["val_acc"] = 100. * correct / total
        metrics["gen_gap"] = metrics["val_loss"] - metrics["train_loss"]

        logger.log_epoch(epoch, metrics)
        snap_monitor.capture_and_save(os.path.join("results", run_id), epoch)
        print(f"Epoch {epoch}: Train Loss={metrics['train_loss']:.4f}, Val Loss={metrics['val_loss']:.4f}")

    logger.save_final()
    act_monitor.remove_hooks()
    rep_monitor.remove_hooks()
    snap_monitor.remove_hooks()

if __name__ == "__main__":
    from config import SMOKE_CONFIG
    train(SMOKE_CONFIG, "smoke_test")
