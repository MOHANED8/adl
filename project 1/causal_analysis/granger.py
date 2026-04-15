from typing import List

import pandas as pd
from sklearn.feature_selection import f_regression


def granger_layer_ranking(dynamics_df: pd.DataFrame, metrics_df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    target = metrics_df[["epoch", "val_loss"]].copy()
    target["future_val_loss"] = target["val_loss"].shift(-1)
    base = dynamics_df.copy()
    out_rows: List[dict] = []
    for layer, layer_df in base.groupby("layer"):
        lagged = layer_df.sort_values("epoch").copy()
        numeric_cols = [c for c in lagged.columns if c not in {"layer", "epoch"}]
        for col in numeric_cols:
            lagged[f"{col}_lag{lag}"] = lagged[col].shift(lag)
        joined = lagged.merge(target[["epoch", "future_val_loss"]], on="epoch", how="inner").dropna()
        if joined.empty:
            continue
        x_cols = [c for c in joined.columns if c.endswith(f"_lag{lag}")]
        x = joined[x_cols]
        y = joined["future_val_loss"]
        f_stat, p_val = f_regression(x, y)
        out_rows.append(
            {
                "layer": layer,
                "mean_f_stat": float(f_stat.mean()),
                "mean_p_value": float(p_val.mean()),
                "n_features": len(x_cols),
            }
        )
    if not out_rows:
        return pd.DataFrame(columns=["layer", "mean_f_stat", "mean_p_value", "n_features"])
    return pd.DataFrame(out_rows).sort_values(["mean_f_stat", "mean_p_value"], ascending=[False, True])
