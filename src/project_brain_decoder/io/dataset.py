import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from project_brain_decoder.io.nwb_loader import load_nwb


def make_windows(neural: np.ndarray, # (T, C)
                 targets: np.ndarray, # (T, out_dim)
                 window_size: int,
                 stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window view over time; target aligned to last timestep of each window."""
    X = np.lib.stride_tricks.sliding_window_view(neural, window_size, axis=0)[::stride]  # (n_windows, C, window_size)
    X = X.transpose(0, 2, 1)  # (n_windows, window_size, C)
    y = targets[window_size - 1::stride][:X.shape[0]]
    return X, y