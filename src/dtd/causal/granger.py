from __future__ import annotations

from typing import Optional

import numpy as np


def granger_matrix(X: np.ndarray, y: np.ndarray, *, maxlag: int = 3) -> np.ndarray:
    """
    Granger causality p-values for each feature -> y.

    Returns pvals shape (F,), using minimum p-value across lags.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as e:
        raise RuntimeError("Missing dependency 'statsmodels'. Install with `pip install statsmodels`.") from e

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    F = X.shape[1]
    pvals = np.ones(F, dtype=float)
    for j in range(F):
        data = np.column_stack([y, X[:, j]])
        # statsmodels expects [y, x] in that order for testing x causes y
        try:
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            pv = min(res[lag][0]["ssr_ftest"][1] for lag in res.keys())
            pvals[j] = float(pv)
        except Exception:
            pvals[j] = 1.0
    return pvals

