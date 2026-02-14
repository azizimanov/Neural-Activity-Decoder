from pathlib import Path
from project_brain_decoder.io.nwb_loader import load_nwb
from project_brain_decoder.io.nwb_loader import get_project_root
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd



root = get_project_root()
raw_data_dir = root / "data" / "raw"
results_dir = root / "results"


def main(path, model, results_dir):
    results = []
    for file_path in path.glob("*.nwb"):
        nwb_file = load_nwb(file_path)
        X = nwb_file["neural_threshold_crossings"]
        y = nwb_file["target_index_velocity"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                            random_state=42, shuffle=False)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X=X_train, y=y_train)
        prediction = model.predict(X=X_test)
        r2 = r2_score(y_pred=prediction, y_true=y_test)
        results.append(tuple([Path(file_path).stem, r2]))
    df = pd.DataFrame(data=results, columns=["Session name", "R2 score"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=results_dir / "ridge_r2.csv", index=False)
    return model



if __name__ == "__main__":
    main(path=raw_data_dir, model=Ridge(), results_dir=results_dir)