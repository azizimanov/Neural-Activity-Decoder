from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tcn import TCN
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from project_brain_decoder.config import get_project_root
from project_brain_decoder.io.nwb_loader import load_nwb
from sklearn.metrics import r2_score

tf.random.set_seed(42)
np.random.seed(42)

data_folder = get_project_root() / "data" / "raw"
batch_size, window_size, input_dim = 128, 15, 192

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
    output_layer = TCN(nb_filters=32, kernel_size=4, dilations=[1, 2, 4, 8, 16],
                       return_sequences=False, dropout_rate=0.3)(input_layer)
    output_layer = Dense(1)(output_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

def main(model):
    # Looping through 12 files with indexing
    files = list(data_folder.glob("*.nwb"))
    train_files = files[:10]
    val_file = files[10]
    test_file = files[11]
    neural_scaler = StandardScaler()
    targets_scaler = StandardScaler()
    neural_list = []
    targets_list = []

    for file in train_files:
        loaded_file = load_nwb(file_path=file)
        spiking = loaded_file["neural_spiking_band"]
        threshold = loaded_file["neural_threshold_crossings"]
        neural = np.concatenate([spiking, threshold], axis=1)
        targets = loaded_file["target_index_velocity"] if loaded_file["target_index_velocity"].ndim == 2 \
            else loaded_file["target_index_velocity"].reshape(-1, 1)
        neural_list.append(neural)
        targets_list.append(targets)
    neural_all = np.concatenate(neural_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    scaled_neural = neural_scaler.fit_transform(neural_all)
    scaled_targets = targets_scaler.fit_transform(targets_all)
    X_train, y_train = make_windows(neural=scaled_neural, targets=scaled_targets, window_size=15, stride=1)


    # X: (n_windows, 15, C), y: (n_windows, ) or (n_windows, d)
    # Validation
    load_val = load_nwb(val_file)
    val_spike = load_val["neural_spiking_band"]
    val_thresh = load_val["neural_threshold_crossings"] # (T, C)
    neural_val = np.concatenate([val_spike, val_thresh], axis=1)
    targets_validation = load_val["target_index_velocity"] # (T,) or (T, d)s
    val_transformed = neural_scaler.transform(neural_val)
    val_targets = targets_scaler.transform(targets_validation if targets_validation.ndim == 2
                                               else targets_validation.reshape(-1, 1))
    X_val, y_val = make_windows(neural=val_transformed, targets=val_targets, window_size=15, stride=1)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    val_pred = model.predict(X_val)
    val_score = r2_score(y_true=y_val, y_pred=val_pred)
    print("Validation score: ", val_score)


    # Test
    load_test = load_nwb(test_file)
    test_spike = load_test["neural_spiking_band"]
    test_thresh = load_test["neural_threshold_crossings"] # (T, C)
    neural_test = np.concatenate([test_spike, test_thresh], axis=1)
    targets_test = load_test["target_index_velocity"] # (T,) or (T, d)
    test_transformed = neural_scaler.transform(neural_test)
    test_targets = targets_scaler.transform(targets_test if targets_test.ndim == 2
                                              else targets_test.reshape(-1, 1))
    X_test, y_test = make_windows(neural=test_transformed, targets=test_targets, window_size=15, stride=1)
    test_pred = model.predict(X_test)
    test_score = r2_score(y_true=y_test, y_pred=test_pred)
    print("Test score: ", test_score)

    # Save val and test scores
    Path(get_project_root() / "results").mkdir(exist_ok=True)
    df = pd.DataFrame(data={"Validation": [val_score], "Test": [test_score]})
    df.to_csv(path_or_buf=get_project_root() / "results" / "tcn_r2.csv", index=False)






if __name__ == "__main__":
    main(model=get_tcn(batch_size=batch_size,
                       window_size=window_size,
                       input_dim=input_dim,
                       TCN=TCN))