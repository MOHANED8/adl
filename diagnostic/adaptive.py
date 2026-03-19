import torch

class RiskBasedEarlyStopping:
    def __init__(self, scorer, threshold=0.8, patience=3):
        self.scorer = scorer
        self.threshold = threshold
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0

    def check(self, epoch, metrics):
        self.scorer.update_history(metrics)
        risk = self.scorer.get_risk_score()
        
        if risk > self.threshold:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Risk-Based Early Stopping triggered at epoch {epoch} (Risk: {risk:.4f})")
                return True
        else:
            self.wait = 0
        return False

class AdaptiveRegularization:
    def __init__(self, optimizer, scorer):
        self.optimizer = optimizer
        self.scorer = scorer

    def adapt(self):
        risk = self.scorer.get_risk_score()
        if risk > 0.5:
            # Increase weight decay if risk is high
            new_wd = 1e-4 * (1.0 + risk)
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = new_wd
            print(f"Adaptive Regularization: Weight decay increased to {new_wd:.6f} due to risk {risk:.4f}")
