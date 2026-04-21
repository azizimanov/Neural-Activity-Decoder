import numpy as np
import tensorflow as tf
from project_brain_decoder.config import get_project_root
from project_brain_decoder.io.dataset import make_windows, make_dataset, count_windows
from project_brain_decoder.models.gru import GRUDecoder
from results import save_r2


tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size, window_size, input_dim = 128, 55, 192
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
n_test_steps = count_windows(test, window_size, stride) // batch_size

# Build datasets
train_ds = make_dataset(train, window_size, stride, input_dim, batch_size, shuffle=True).repeat()
val_ds = make_dataset(val, window_size, stride, input_dim, batch_size).repeat()
test_ds = make_dataset(test, window_size, stride, input_dim, batch_size).repeat()

# Train
decoder = GRUDecoder(window_size, input_dim, batch_size, stride)
decoder.fit(train_ds=train_ds, val_ds=val_ds, n_train_steps=n_train_steps, n_val_steps=n_val_steps)