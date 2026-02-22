from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Input, Model
from tcn import TCN
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from project_brain_decoder.config import get_project_root
from src.project_brain_decoder.io.nwb_loader import load_nwb


data_folder = get_project_root() / "data" / "raw"
batch_size, window_size, input_dim = 64, 15, 96

def make_windows(neural: np.array, # shape(T, C) - time * channels
                 targets: np.array, # shape(T,) or (T, out_dim)
                 window_size: int,
                 stride: int=1) -> tuple[np.array, np.array]:
    """Slice into (window_size, C) windows; targets aligned to last timestep of each window"""
    T, C = neural.shape
    n_windows = (T - window_size) // stride + 1
    X_train = np.stack([neural[i : i + window_size] for i in range(0, T - window_size + 1, stride)])
    # target for each window = value at the end of the window
    y_train = targets[window_size - 1 :: stride][:n_windows]
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
    return X_train, y_train




def main(model, folder, scaler):
    for i, file in enumerate(data_folder.glob("*.nwb")):
        if i==10:
            break
        loaded_file = load_nwb(file_path=file)
        neural = loaded_file["neural_threshold_crossings"] # (T, C)
        targets = loaded_file["target_index_velocity"] # (T,) or (T, d)
        X, y = make_windows(neural=neural, targets=targets, window_size=15, stride=1)
        # X: (n_windows, 15, C), y: (n_windows, ) or (n_windows, d)

























if __name__ == "__main__":
    main(model=TCN(), folder=data_folder, scaler=StandardScaler)