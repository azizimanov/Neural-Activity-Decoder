import numpy as np
import tensorflow as tf
from project_brain_decoder.config import get_project_root
from project_brain_decoder.io.dataset import make_windows, make_dataset, count_windows
from project_brain_decoder.models.tcn import TCNDecoder
from results import save_r2

tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size, window_size, input_dim = 128, 30, 192
stride = 5


# Load and split files
folder = get_project_root() / "data" / "raw"
files = list(folder.glob("*.nwb"))
train = files[:187]
val = files[187:249]
test = files[249:]

# Count steps
n_train_steps = count_windows(train, window_size, stride) // batch_size
n_val_steps = count_windows(val, window_size, stride) // batch_size

# Build generator-based datasets with per-session scaling
train_ds = make_dataset(train, window_size, stride, input_dim, batch_size, shuffle=True).repeat()
val_ds = make_dataset(val, window_size, stride, input_dim, batch_size).repeat()

# Train TCN
decoder = TCNDecoder(window_size, input_dim, batch_size, stride)
decoder.fit(train_ds=train_ds, val_ds=val_ds, n_train_steps=n_train_steps, n_val_steps=n_val_steps)



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