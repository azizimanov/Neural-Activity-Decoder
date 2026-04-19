import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Layer
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class PositionalEncoder(Layer):
    """Injects sinusoidal position information into the input sequence."""
    def __init__(self, window_size, d_model):
        super().__init__()
        pos = np.arange(window_size)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = pos / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rates_2 = angle_rates.copy()
        angle_rates_2[:, 0::2] = np.sin(angle_rates[:, 0::2])  # even indices -> sin
        angle_rates_2[:, 1::2] = np.cos(angle_rates[:, 1::2])  # odd indices -> cos
        self.pe = tf.constant(angle_rates_2[np.newaxis, :, :], dtype=tf.float32)  # (1, window_size, d_model)

    def call(self, x):
        return x + self.pe  # broadcast across batch



class TransformerDecoder:
    """Transformer-based neural decoder for 2D motor velocity prediction."""

    def __init__(self, window_size, input_dim, batch_size, stride):
        self.window_size = window_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stride = stride
        self.model = self._build_model()


