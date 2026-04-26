import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tcn import TCN
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from project_brain_decoder.config import get_project_root
from project_brain_decoder.io.dataset import make_windows
from project_brain_decoder.io.nwb_loader import load_nwb
from project_brain_decoder.models.kalman import KalmanDecoder
from project_brain_decoder.models.ridge import RidgeDecoder
from project_brain_decoder.models.transformer import PositionalEncoder



# Hyperparameters
WINDOW_SIZE, INPUT_DIM, BATCH_SIZE, STRIDE = 55, 192, 128, 5
TCN_WINDOW = 55


def get_session_data(file):
    """Loads neural and target arrays from an NWB session"""
    session = load_nwb(file)
    neural = np.concatenate([session["neural_spiking_band"], session["neural_threshold_crossings"]], axis=1)
    targets = np.column_stack([session["target_index_velocity"], session["target_mrs_velocity"]])
    return neural, targets


def evaluate_keras(model, test_files, window_size):
    """Per-session evaluation for Keras models (GRU, Transformer, TCN)"""
    per_session_r2 = []
    for file in test_files:
        neural, targets = get_session_data(file)
        neural_scaled = StandardScaler().fit_transform(neural)
        targets_scaled = StandardScaler().fit_transform(targets)

        X, y = make_windows(neural_scaled, targets_scaled, window_size, STRIDE)
        y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        r2 = r2_score(y, y_pred, multioutput="raw_values")
        per_session_r2.append((file.stem, r2[0], r2[1]))
    return per_session_r2


def evaluate_ridge(decoder, test_files):
    """Per-session evaluation for Ridge"""
    per_session_r2 = []
    for file in test_files:
        neural, targets = get_session_data(file)
        neural_scaled = StandardScaler().fit_transform(neural)
        targets_scaled = StandardScaler().fit_transform(targets)
        y_pred = decoder.model.predict(neural_scaled)
        r2 = r2_score(targets_scaled, y_pred, multioutput="raw_values")
        per_session_r2.append((file.stem, r2[0], r2[1]))
    return per_session_r2


def evaluate_kalman(decoder, test_files):
    """Per-session evaluation for Kalman"""
    per_session_r2 = []
    for file in test_files:
        d = load_nwb(file)
        neural = np.concatenate([d["neural_spiking_band"], d["neural_threshold_crossings"]], axis=1)
        state = np.column_stack([
            d["target_index_position"], d["target_mrs_position"],
            d["target_index_velocity"], d["target_mrs_velocity"],
        ])
        n_scaled = StandardScaler().fit_transform(neural)
        s_scaled = StandardScaler().fit_transform(state)
        z = decoder.pca.transform(n_scaled)

        decoder.kf.x = np.zeros((decoder.dim_state, 1))
        decoder.kf.x[-1] = 1.0
        decoder.kf.P = np.eye(decoder.dim_state)

        preds = []
        for t in range(len(z)):
            decoder.kf.predict()
            decoder.kf.update(z[t].reshape(-1, 1))
            preds.append(decoder.kf.x.copy().flatten())
        preds = np.array(preds)
        r2 = r2_score(s_scaled[:, 2:4], preds[:, 2:4], multioutput="raw_values")
        per_session_r2.append((file.stem, r2[0], r2[1]))
    return per_session_r2


def get_predictions_one_session(model_dict, file):
    """Returns dict of {model_name: (y_true, y_pred)} for a single session — used for trace plots"""
    neural, targets = get_session_data(file)
    neural_scaled = StandardScaler().fit_transform(neural)
    targets_scaled = StandardScaler().fit_transform(targets)
    preds = {}

    # Keras models
    for name, (model, win) in model_dict["keras"].items():
        X, y = make_windows(neural_scaled, targets_scaled, win, STRIDE)
        y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        # Align: predictions correspond to time indices [win-1::STRIDE]
        preds[name] = (y, y_pred)

    # Ridge
    if "ridge" in model_dict:
        y_pred = model_dict["ridge"].model.predict(neural_scaled)
        preds["ridge"] = (targets_scaled, y_pred)

    # Kalman
    if "kalman" in model_dict:
        kal = model_dict["kalman"]
        d = load_nwb(file)
        state = np.column_stack([
            d["target_index_position"], d["target_mrs_position"],
            d["target_index_velocity"], d["target_mrs_velocity"],
        ])
        s_scaled = StandardScaler().fit_transform(state)
        z = kal.pca.transform(neural_scaled)
        kal.kf.x = np.zeros((kal.dim_state, 1))
        kal.kf.x[-1] = 1.0
        kal.kf.P = np.eye(kal.dim_state)
        kal_preds = []
        for t in range(len(z)):
            kal.kf.predict()
            kal.kf.update(z[t].reshape(-1, 1))
            kal_preds.append(kal.kf.x.copy().flatten())
        kal_preds = np.array(kal_preds)
        preds["kalman"] = (s_scaled[:, 2:4], kal_preds[:, 2:4])

    return preds


def plot_comparison_bar(df, results_dir):
    """Bar chart comparing mean R² across models"""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = df["Model"].values
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, df["Index vel. R2 Score"], width, label="Index velocity")
    ax.bar(x + width/2, df["MRS vel. R2 Score"], width, label="MRS velocity")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("R²")
    ax.set_title("Cross-session decoder comparison")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close()


def plot_per_session_box(per_session_dfs, results_dir):
    """Boxplot of per-session R² across models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dim, title in zip(axes, ["index_r2", "mrs_r2"], ["Index velocity", "MRS velocity"]):
        data = [df[dim].values for df in per_session_dfs.values()]
        ax.boxplot(data, labels=list(per_session_dfs.keys()))
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.axhline(0, color="red", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "per_session_distribution.png", dpi=150)
    plt.close()


def plot_traces(preds, session_name, results_dir, n_seconds=30, fs=50):
    """Predicted vs true velocity traces for one session"""
    n_samples = n_seconds * fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for dim, (ax, title) in enumerate(zip(axes, ["Index velocity", "MRS velocity"])):
        truth_plotted = False
        for name, value in preds.items():
            if name == "truth":
                continue
            y_true, y_pred = value
            t = np.arange(min(n_samples, len(y_pred))) / fs

            # Plot ground truth once (use first model's y_true)
            if not truth_plotted:
                ax.plot(t, y_true[:n_samples, dim], color="black",
                        linewidth=1.5, label="Truth", alpha=0.7)
                truth_plotted = True

            ax.plot(t, y_pred[:n_samples, dim], linewidth=1, alpha=0.8, label=name)

        ax.set_ylabel(f"{title}\n(scaled)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Predicted vs. true velocity — {session_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"traces_{session_name}.png", dpi=150)
    plt.close()

def main():
    root = get_project_root()
    models_dir = root / "saved_models"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load test files (same split as training)
    files = sorted((root / "data" / "raw").glob("*.nwb"))
    test_files = files[249:]

    # Load all models
    gru = load_model(models_dir / "gru.keras")
    transformer = load_model(models_dir / "transformer.keras",
                             custom_objects={"PositionalEncoder": PositionalEncoder})

    tcn = load_model(models_dir / "tcn.keras", custom_objects={"TCN": TCN})

    ridge = RidgeDecoder()
    ridge.load(models_dir / "ridge.pkl")

    kalman = KalmanDecoder()
    kalman.load(models_dir / "kalman.pkl")

    # Per-session R²
    print("Evaluating GRU..."); gru_r2 = evaluate_keras(gru, test_files, WINDOW_SIZE)
    print("Evaluating Transformer..."); tr_r2 = evaluate_keras(transformer, test_files, WINDOW_SIZE)
    print("Evaluating TCN..."); tcn_r2 = evaluate_keras(tcn, test_files, TCN_WINDOW)
    print("Evaluating Ridge..."); ridge_r2 = evaluate_ridge(ridge, test_files)
    print("Evaluating Kalman..."); kalman_r2 = evaluate_kalman(kalman, test_files)

    per_session_dfs = {}
    for name, scores in [("gru", gru_r2), ("transformer", tr_r2), ("tcn", tcn_r2),
                         ("ridge", ridge_r2), ("kalman", kalman_r2)]:
        df = pd.DataFrame(scores, columns=["session", "index_r2", "mrs_r2"])
        df.to_csv(results_dir / f"per_session_r2_{name}.csv", index=False)
        per_session_dfs[name] = df

    # Comparison plots
    summary = pd.read_csv(results_dir / "r2_scores.csv")
    plot_comparison_bar(summary, results_dir)
    plot_per_session_box(per_session_dfs, results_dir)

    # Trace plot for one example session
    example_file = test_files[0]
    model_dict = {
        "keras": {"gru": (gru, WINDOW_SIZE), "transformer": (transformer, WINDOW_SIZE), "tcn": (tcn, TCN_WINDOW)},
        "ridge": ridge,
        "kalman": kalman,
    }
    preds = get_predictions_one_session(model_dict, example_file)
    plot_traces(preds, example_file.stem, results_dir)

    print(f"\nDone. Results saved to {results_dir}")


if __name__ == "__main__":
    main()