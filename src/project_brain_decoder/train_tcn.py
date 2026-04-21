import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from tcn import TCN
from project_brain_decoder.config import get_project_root
from project_brain_decoder.io.dataset import make_dataset, make_windows, count_windows
from results import save_r2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size, window_size, input_dim = 128, 30, 192
stride = 5


def get_tcn(window_size, input_dim):
    input_layer = Input(shape=(window_size, input_dim))
    tcn_layer = TCN(nb_filters=32, kernel_size=4, dilations=[1, 2, 4, 8, 16],
                       return_sequences=False, dropout_rate=0.3)(input_layer)
    output_layer = Dense(2)(tcn_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model


def main():
    folder = get_project_root() / "data" / "raw"
    files = list(folder.glob("*.nwb"))
    train_files = files[:187]
    val_files = files[187:249]
    test_files = files[249:]

    # Count steps
    n_train_steps = count_windows(train_files, window_size, stride) // batch_size
    n_val_steps = count_windows(val_files, window_size, stride) // batch_size

    # Build generator-based datasets with per-session scaling
    train_ds = make_dataset(train_files, window_size, stride, input_dim, batch_size, shuffle=True).repeat()
    val_ds = make_dataset(val_files, window_size, stride, input_dim, batch_size).repeat()

    # Train
    model = get_tcn(window_size, input_dim)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    model.fit(train_ds, epochs=20, steps_per_epoch=n_train_steps,
              validation_data=val_ds, validation_steps=n_val_steps,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True), reduce_lr])

    # Test — per-session scaling and evaluation
    from project_brain_decoder.io.nwb_loader import load_nwb
    r2_list = []
    for file in test_files:
        session = load_nwb(file)
        neural = np.concatenate([session["neural_spiking_band"], session["neural_threshold_crossings"]], axis=1)
        targets = np.column_stack([session["target_index_velocity"], session["target_mrs_velocity"]])

        neural_scaler = StandardScaler()
        targets_scaler = StandardScaler()
        neural_scaled = neural_scaler.fit_transform(neural)
        targets_scaled = targets_scaler.fit_transform(targets)

        X_test, y_test = make_windows(neural_scaled, targets_scaled, window_size, stride)
        y_pred = model.predict(X_test, batch_size=batch_size)
        r2 = r2_score(y_test, y_pred, multioutput="raw_values")
        r2_list.append(r2)

    mean_r2 = np.mean(r2_list, axis=0)
    print(f"Index vel. R²: {mean_r2[0]:.4f}. MRS vel. R²: {mean_r2[1]:.4f}")

    # Save scores
    save_r2.get_scores(model="tcn", score1=mean_r2[0], score2=mean_r2[1])


if __name__ == "__main__":
    main()