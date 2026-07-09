import numpy as np
import torch
from torch.utils.data import Dataset


class FBmDataset(Dataset):
    """
    
    Create a PyTorch Dataset.

    Each item is returned as a tensor of shape (T, 1) (sequence, features)

    Also the target is value of H (Hurst)

    """

    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 normalize: str | None = "zscore"):
        
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, T)
            Raw trajectories.
        y : np.ndarray, shape (N,)
            Hurst exponent targets.
        normalize : {"zscore", "increments", None}
        """
        # check inputs and targets have valid size (1 to 1 mapping)
        assert X.shape[0] == y.shape[0]
        self.normalize = normalize
        self.X = self._preprocess(X.astype(np.float32))
        self.y = y.astype(np.float32)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:

        if self.normalize == "zscore":
            # simple normalization
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True) + 1e-8
            X = (X - mean) / std


        elif self.normalize == "increments":

            # here calculating fGN from the path 
            # normalize the noise not path
            # Note here the feature is noise fGN not Brownina motion fBM.
            # This is regularly used in Time Series Analysis

            inc = np.diff(X, axis=1, prepend=X[:, :1])
            mean = inc.mean(axis=1, keepdims=True)
            std = inc.std(axis=1, keepdims=True) + 1e-8
            X = (inc - mean) / std


        elif self.normalize is None:
            pass

        else:
            raise ValueError(f"Unknown normalize option: {self.normalize}")
        return X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # single trajectory instance 
        # Sample give a torch tensor.
        x = torch.from_numpy(self.X[idx]).unsqueeze(-1)  # (T, 1)
        y = torch.tensor(self.y[idx])
        return x, y


# dataset for the simple regressions
def train_val_test_split(X: np.ndarray,
                         y: np.ndarray, 
                         val_frac: float = 0.15,
                         test_frac: float = 0.15, 
                         seed: int = 0,
                         stratify_bins: int = 10):
    
    """
    
    Split (X, y) into train/val/test with approximately uniform
    coverage of H across splits, via stratified binning of H.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)

    """
    rng = np.random.default_rng(seed)
    n = len(y)

    bins = np.linspace(y.min(), y.max() + 1e-8, stratify_bins + 1)
    bin_idx = np.digitize(y, bins) - 1

    train_idx, val_idx, test_idx = [], [], []

    for b in range(stratify_bins):


        idx = np.where(bin_idx == b)[0]
        rng.shuffle(idx)
        n_b = len(idx)
        n_val = int(round(n_b * val_frac))
        n_test = int(round(n_b * test_frac))
        val_idx.extend(idx[:n_val])
        test_idx.extend(idx[n_val:n_val + n_test])
        train_idx.extend(idx[n_val + n_test:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)


    return (

        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),

    )
