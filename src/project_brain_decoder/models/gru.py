import numpy as np
from keras.layers import Input, Dense, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.project_brain_decoder.io.nwb_loader import load_nwb


class GRUDecoder:
    """GRU-based neural decoder for 2D motor velocity prediction"""

    def __init__(self, window_size, input_dim, batch_size, stride):
        self.window_size = window_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stride = stride
        self.model = self._build_model()

    def _build_model(self):
        """Builds a two-layer GRU followed by a Dense projection to 2D velocity"""
        input_layer = Input(shape=(self.window_size, self.input_dim))
        gru_1 = GRU(units=128, dropout=0.3, return_sequences=True)(input_layer)
        gru_2 = GRU(units=64, dropout=0.5)(gru_1)
        output = Dense(units=2)(gru_2)

        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
        return model

    def fit(self, train_ds, val_ds, n_train_steps, n_val_steps):
        """Trains the model on the full dataset using generator-based pipelines"""
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        self.model.fit(train_ds, epochs=20, steps_per_epoch=n_train_steps,
                       validation_data=val_ds, validation_steps=n_val_steps,
                       callbacks=[EarlyStopping(patience=3, restore_best_weights=True), reduce_lr])

    def fine_tune(self, test_files, make_windows):
        """Freezes GRU layers and fine-tunes the output Dense layer per test session"""
        for layer in self.model.layers:
            if isinstance(layer, GRU):
                layer.trainable = False
        self.model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")

        # Save base weights to reset between sessions
        dense_layer = [l for l in self.model.layers if isinstance(l, Dense)][-1]
        base_dense_weights = dense_layer.get_weights()


