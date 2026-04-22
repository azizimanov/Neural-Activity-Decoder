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
