import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from project_brain_decoder.io.nwb_loader import load_nwb

class RidgeDecoder:
    """Ridge regression-based neural decoder for 2D motor velocity prediction"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, train_files):
        """Fits Ridge regression on pooled, per-session-scaled neural and velocity data"""
        neural_list, targets_list = [], []
        for file in train_files:
            session = load_nwb(file)
            neural = np.concatenate([session["neural_spiking_band"], session["neural_threshold_crossings"]], axis=1)
            targets = np.column_stack([session["target_index_velocity"], session["target_mrs_velocity"]])

            # Per-session scaling (consistent with other decoders)
            neural = StandardScaler().fit_transform(neural)
            targets = StandardScaler().fit_transform(targets)

            neural_list.append(neural)
            targets_list.append(targets)

        X = np.concatenate(neural_list, axis=0)
        y = np.concatenate(targets_list, axis=0)
        self.model.fit(X, y)

    def evaluate(self, test_files):
        """Evaluates per-session on test files with independent scaling"""
        r2_list = []
        for file in test_files:
            session = load_nwb(file)
            neural = np.concatenate([session["neural_spiking_band"], session["neural_threshold_crossings"]], axis=1)
            targets = np.column_stack([session["target_index_velocity"], session["target_mrs_velocity"]])

            neural_scaled = StandardScaler().fit_transform(neural)
            targets_scaled = StandardScaler().fit_transform(targets)

            y_pred = self.model.predict(neural_scaled)
            r2 = r2_score(targets_scaled, y_pred, multioutput="raw_values")
            r2_list.append(r2)

        return np.mean(r2_list, axis=0)

    def save(self, path):
        """Saves the trained model to disk"""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "alpha": self.alpha}, f)

    def load(self, path):
        """Loads a trained model from disk"""
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.model = d["model"]
        self.alpha = d["alpha"]