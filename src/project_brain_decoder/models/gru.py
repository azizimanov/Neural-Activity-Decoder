import numpy as np
from keras.layers import Input, Dense, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.project_brain_decoder.io.nwb_loader import load_nwb

