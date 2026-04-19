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


    def _build_model(self):
        """Builds the transformer architecture with one attention block and a feed-forward block."""
        input_layer = Input(shape=(self.window_size, self.input_dim))
        pos_encoded = PositionalEncoder(self.window_size, self.input_dim)(input_layer)
        drop_out_1 = Dropout(0.1)(pos_encoded)

        # Multi-head self-attention + residual connection
        attention_1 = MultiHeadAttention(num_heads=4, key_dim=48)(drop_out_1, drop_out_1)
        drop_out_2 = Dropout(0.1)(attention_1)
        skip_1 = LayerNormalization()(drop_out_2 + pos_encoded)

        # Feed-forward block + residual connection
        dense_1 = Dense(units=128, activation="relu")(skip_1)
        drop_out_3 = Dropout(0.1)(dense_1)
        dense_2 = Dense(units=192)(drop_out_3)
        skip_2 = LayerNormalization()(dense_2 + skip_1)

        # Pool across timesteps and project to 2D velocity output
        avg_pool = GlobalAveragePooling1D()(skip_2)
        output = Dense(units=2)(avg_pool)

        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
        return model


    def fit(self, train_ds, val_ds, n_train_steps, n_val_steps):
        """Trains the model on the full dataset using generator-based pipelines."""
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        self.model.fit(train_ds, epochs=20, steps_per_epoch=n_train_steps,
                       validation_data=val_ds, validation_steps=n_val_steps,
                       callbacks=[EarlyStopping(patience=3, restore_best_weights=True), reduce_lr])


    