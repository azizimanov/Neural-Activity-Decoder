import numpy as np
from project_brain_decoder.config import get_project_root
from project_brain_decoder.models.ridge import RidgeDecoder
from results import save_r2

np.random.seed(42)

# Load and split files
folder = get_project_root() / "data" / "raw"
files = list(folder.glob("*.nwb"))
train = files[:187]
val = files[187:249]
test = files[249:]


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
        results.append(tuple([Path(file_path).stem, f"{r2}"]))
    df = pd.DataFrame(data=results, columns=["Session name", "R2 score"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=results_dir / "ridge_r2.csv", index=False)
    return model



if __name__ == "__main__":
    main(path=data_dir, model=Ridge(), results_dir=results_dir)