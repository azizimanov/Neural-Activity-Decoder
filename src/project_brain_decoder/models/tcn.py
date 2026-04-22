import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tcn import TCN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.project_brain_decoder.io.nwb_loader import load_nwb


class TCNDecoder:
    """Temporal Convolutional Network-based neural decoder for 2D motor velocity prediction"""

    def __init__(self, window_size, input_dim, batch_size, stride):
        self.window_size = window_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stride = stride
        self.model = self._build_model()


    def _build_model(self):
        """Builds a TCN with exponentially dilated convolutions and a Dense projection to 2D velocity"""
        input_layer = Input(shape=(self.window_size, self.input_dim))
        tcn_layer = TCN(nb_filters=32, kernel_size=4, dilations=[1, 2, 4, 8, 16],
                        return_sequences=False, dropout_rate=0.3)(input_layer)
        output = Dense(units=2)(tcn_layer)

        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
        return model

    def fit(self, train_ds, val_ds, n_train_steps, n_val_steps):
        """Trains the model on the full dataset using generator-based pipelines"""
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        self.model.fit(train_ds, epochs=20, steps_per_epoch=n_train_steps,
                       validation_data=val_ds, validation_steps=n_val_steps,
                       callbacks=[EarlyStopping(patience=5, restore_best_weights=True), reduce_lr])
