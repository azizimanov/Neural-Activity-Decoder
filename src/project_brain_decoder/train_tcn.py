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
    X = np.stack([neural[i : i + window_size] for i in range(0, T - window_size + 1, stride)])
    # target for each window = value at the end of the window
    y = targets[window_size - 1 :: stride][:n_windows]
    # if y.ndim == 1:
    #     y = y[:, np.newaxis]
    return X, y


def get_tcn(batch_size, window_size, input_dim, TCN):
    input_layer = Input(batch_shape=(batch_size, window_size, input_dim))
    output_layer = TCN(return_sequences=False)(input_layer)
    output_layer = Dense(1)(output_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer="adam", loss="mse")
    return model

def main(model):
    # Looping through 12 files with indexing
    files = list(data_folder.glob("*.nwb"))
    train_files = files[:10]
    val_file = files[10]
    test_file = files[11]

    for file in train_files:
        loaded_file = load_nwb(file_path=file)
        neural = loaded_file["neural_threshold_crossings"] # (T, C)
        targets = loaded_file["target_index_velocity"] # (T,) or (T, d)
        neural_scaler = StandardScaler()
        targets_scaler = StandardScaler()
        train_neural = neural_scaler.partial_fit(neural)
        train_targets = targets_scaler.partial_fit(targets if targets.ndim == 2
                                                     else targets.reshape(-1, 1))
        X_train, y_train = make_windows(neural=train_neural, targets=train_targets,
                                        window_size=15, stride=1)
        # X: (n_windows, 15, C), y: (n_windows, ) or (n_windows, d)
        if i==10:
            neural_validation = loaded_file["neural_threshold_crossings"] # (T, C)
            targets_validation = loaded_file["target_index_velocity"] # (T,) or (T, d)s
            val_neural = neural_scaler.transform(neural_validation)
            val_targets = targets_scaler.transform(targets_validation if targets_validation.ndim == 2
                                                   else targets_validation.reshape(-1, 1))
        elif i==11:
            neural_test = loaded_file["neural_threshold_crossings"] # (T, C)
            targets_test = loaded_file["target_index_velocity"] # (T,) or (T, d)
            test_neural = neural_scaler.transform(neural_test)
            targets_neural = targets_scaler.transform(targets_test if targets_test.ndim == 2
                                                      else targets_test.reshape(-1, 1))
            pred_velocity = model.predict(test_neural)
        model.fit(X_train, y_train, epochs=10)


if __name__ == "__main__":
    main(model=get_tcn(batch_size=batch_size,
                       window_size=window_size,
                       input_dim=input_dim,
                       TCN=TCN))