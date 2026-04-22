import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import r2_score
from src.project_brain_decoder.io.nwb_loader import load_nwb


class KalmanDecoder:
    """Kalman filter-based neural decoder for 2D motor velocity prediction"""

    def __init__(self, n_components=50, dim_state=5):
        self.n_components = n_components
        self.dim_state = dim_state  # [pos_idx, pos_mrs, vel_idx, vel_mrs, 1]
        self.pca = IncrementalPCA(n_components=n_components)
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self.kf = None


    @staticmethod
    def _load_session(file):
        """Loads one NWB file, returns per-session-scaled neural and bias-augmented state"""
        d = load_nwb(file)
        neural = np.concatenate([d["neural_spiking_band"], d["neural_threshold_crossings"]], axis=1)
        state = np.column_stack([
            d["target_index_position"], d["target_mrs_position"],
            d["target_index_velocity"], d["target_mrs_velocity"],
        ])
        n_scaled = StandardScaler().fit_transform(neural)
        s_scaled = StandardScaler().fit_transform(state)
        s_aug = np.column_stack([s_scaled, np.ones(len(s_scaled))])  # (T, 5)
        return n_scaled, s_aug


    def _fit_pca(self, train_files):
        """Pass 1: fits IncrementalPCA one session at a time"""
        for file in train_files:
            n_scaled, _ = self._load_session(file)
            self.pca.partial_fit(n_scaled)
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")


    def _fit_transition_observation(self, train_files):
        """Pass 2: accumulates sufficient statistics for F (transition) and H (observation)"""
        F_XtX = np.zeros((self.dim_state, self.dim_state))
        F_XtY = np.zeros((self.dim_state, self.dim_state))
        H_AtA = np.zeros((self.dim_state, self.dim_state))
        H_AtB = np.zeros((self.dim_state, self.n_components))

        for file in train_files:
            n_scaled, s_aug = self._load_session(file)
            z = self.pca.transform(n_scaled)

            # F: per-session consecutive pairs (no cross-session contamination)
            prev, curr = s_aug[:-1], s_aug[1:]
            F_XtX += prev.T @ prev
            F_XtY += prev.T @ curr

            # H: all timesteps
            H_AtA += s_aug.T @ s_aug
            H_AtB += s_aug.T @ z

        self.F = np.linalg.solve(F_XtX, F_XtY).T
        self.F[-1, :] = 0.0
        self.F[-1, -1] = 1.0  # bias stays constant
        self.H = np.linalg.solve(H_AtA, H_AtB).T  # (n_components, dim_state)


    def _fit_noise(self, train_files):
        """Pass 3: computes Q and R from residuals"""
        Q_sum = np.zeros((self.dim_state, self.dim_state))
        R_sum = np.zeros(self.n_components)
        n_F, n_H = 0, 0

        for file in train_files:
            n_scaled, s_aug = self._load_session(file)
            z = self.pca.transform(n_scaled)

            prev, curr = s_aug[:-1], s_aug[1:]
            res_F = curr - (self.F @ prev.T).T
            Q_sum += res_F.T @ res_F
            n_F += len(res_F)

            res_H = z - (self.H @ s_aug.T).T
            R_sum += np.sum(res_H ** 2, axis=0)
            n_H += len(res_H)

        self.Q = Q_sum / n_F
        self.Q[-1, :] = 0.0
        self.Q[:, -1] = 0.0
        self.Q[-1, -1] = 1e-12
        self.R = np.diag(R_sum / n_H)


    def _build_filter(self):
        """Assembles the Kalman filter from fitted parameters"""
        self.kf = KalmanFilter(dim_x=self.dim_state, dim_z=self.n_components)
        self.kf.x = np.zeros((self.dim_state, 1))
        self.kf.x[-1] = 1.0  # bias term
        self.kf.P = np.eye(self.dim_state)
        self.kf.F = self.F
        self.kf.H = self.H
        self.kf.Q = self.Q
        self.kf.R = self.R


    def fit(self, train_files):
        """Runs the three-pass training procedure and assembles the Kalman filter"""
        self._fit_pca(train_files)
        self._fit_transition_observation(train_files)
        self._fit_noise(train_files)
        self._build_filter()


    def evaluate(self, val_files):
        """Runs the Kalman filter on val sessions and returns mean R² for velocity targets"""
        all_y_true, all_y_pred = [], []

        for file in val_files:
            d = load_nwb(file)
            neural = np.concatenate([d["neural_spiking_band"], d["neural_threshold_crossings"]], axis=1)
            state = np.column_stack([
                d["target_index_position"], d["target_mrs_position"],
                d["target_index_velocity"], d["target_mrs_velocity"],
            ])

            # Per-session scaling (same as training)
            n_scaled = StandardScaler().fit_transform(neural)
            s_scaled = StandardScaler().fit_transform(state)
            z = self.pca.transform(n_scaled)

            # Ground-truth: scaled velocity (indices 2,3)
            y_true = s_scaled[:, 2:4]

            # Reset filter state for each session
            self.kf.x = np.zeros((self.dim_state, 1))
            self.kf.x[-1] = 1.0
            self.kf.P = np.eye(self.dim_state)

            preds = []
            for t in range(len(z)):
                self.kf.predict()
                self.kf.update(z[t].reshape(-1, 1))
                preds.append(self.kf.x.copy().flatten())

            preds = np.array(preds)
            all_y_true.append(y_true)
            all_y_pred.append(preds[:, 2:4])

        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        return r2_score(y_true=all_y_true, y_pred=all_y_pred, multioutput="raw_values")