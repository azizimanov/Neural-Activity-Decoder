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


def make_dataset(file_list: list,
                 window_size: int,
                 stride: int,
                 input_dim: int,
                 batch_size: int,
                 shuffle: bool = False) -> tf.data.Dataset:
    """Generator-based tf.data pipeline; scales each session independently."""
    def generator():
        files = file_list.copy()
        if shuffle:
            np.random.shuffle(files)
        for file in files:
            session = load_nwb(file_path=file)
            neural = np.concatenate([session["neural_spiking_band"], session["neural_threshold_crossings"]], axis=1)
            targets = np.column_stack([session["target_index_velocity"], session["target_mrs_velocity"]])
            # Fit scaler per session
            neural_scaler = StandardScaler()
            targets_scaler = StandardScaler()
            X_scaled = neural_scaler.fit_transform(neural)
            y_scaled = targets_scaler.fit_transform(targets)
            X_w, y_w = make_windows(X_scaled, y_scaled, window_size, stride)
            for i in range(len(X_w)):
                yield X_w[i], y_w[i]

    ds = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(tf.TensorSpec(shape=(window_size, input_dim), dtype=tf.float32),
                          tf.TensorSpec(shape=(2,), dtype=tf.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds