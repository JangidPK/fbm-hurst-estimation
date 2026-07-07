"""
Handcrafted feature engineering for the baseline Hurst-exponent 
regressors

Features extracted from each raw trajectory:
    - variance of increments
    - mean absolute increment
    - autocorrelation of increments at lags 1..10
    - rescaled range (R/S) statistic
    - log-log slope estimate of R/S vs window size (a classic Hurst estimator)
"""

import numpy as np


def _autocorr(x: np.ndarray, 
              
              lag: int) -> float:
    
    x = x - x.mean()
    n = len(x)
    if lag >= n:
        return 0.0
    num = np.sum(x[:n - lag] * x[lag:])
    den = np.sum(x ** 2) + 1e-12
    return num / den


def _rs_statistic(series: np.ndarray) -> float:

    """Classic rescaled range (R/S) statistic for a single window."""

    mean = series.mean()
    dev = np.cumsum(series - mean)
    R = dev.max() - dev.min()
    S = series.std() + 1e-12 # avoid zeroooo

    return R / S


def _rs_loglog_slope(x: np.ndarray, 
                     min_window: int = 8, 
                     n_windows: int = 10) -> float:
    
    """Estimate H via the slope of log(R/S) vs log(window size), computed
    over several window sizes (classic Hurst R/S analysis)."""

    n = len(x)
    max_window = n // 2
    if max_window <= min_window:
        return 0.5

    sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window),n_windows).astype(int))
    rs_vals = []
    valid_sizes = []
    for w in sizes:
        if w < 2:
            continue
        n_segments = n // w
        if n_segments < 1:
            continue
        segments = x[: n_segments * w].reshape(n_segments, w)
        rs = np.mean([_rs_statistic(seg) for seg in segments])
        if rs > 0:
            rs_vals.append(rs)
            valid_sizes.append(w)

    if len(valid_sizes) < 2:
        return 0.5

    log_sizes = np.log(valid_sizes)
    log_rs = np.log(rs_vals)
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def extract_features(X: np.ndarray, 
                     n_lags: int = 10) -> np.ndarray:

    """Extract handcrafted features from a batch of trajectories.

    Parameters
    ----------
    X : np.ndarray, shape (N, T)
        Raw fBm trajectories.
    n_lags : int
        Number of autocorrelation lags to include.

    Returns
    -------
    features : np.ndarray, shape (N, 3 + n_lags)
    """

    N = X.shape[0]
    n_feats = 3 + n_lags
    features = np.zeros((N, n_feats))

    for i in range(N):
        path = X[i]
        increments = np.diff(path)

        var_inc = np.var(increments)
        mean_abs_inc = np.mean(np.abs(increments))
        rs_slope = _rs_loglog_slope(path)

        # these are important to capture temporal memory 
        acfs = [_autocorr(increments, lag) for lag in range(1, n_lags + 1)]

        features[i, 0] = var_inc
        features[i, 1] = mean_abs_inc
        features[i, 2] = rs_slope
        features[i, 3:] = acfs

    return features


FEATURE_NAMES = (
    ["var_increments", "mean_abs_increments", "rs_loglog_slope"]
    + [f"acf_lag_{k}" for k in range(1, 11)]
)
