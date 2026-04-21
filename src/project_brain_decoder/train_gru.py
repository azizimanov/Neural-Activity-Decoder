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