import pandas as pd
from keras.src.backend.jax.numpy import empty

from project_brain_decoder.config import get_project_root

def get_scores(model, score1, score2):
    try:
        csv = pd.read_csv(get_project_root() / "results" / "r2_scores.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        csv = pd.DataFrame(columns=["Model", "Index vel. R2 Score", "MRS vel. R2 Score"])

    if model not in csv["Model"].values:
        new_row = pd.DataFrame({"Model": [model],
                                "Index vel. R2 Score": [score1],
                                "MRS vel. R2 Score": [score2]})
        csv = new_row if csv.empty else pd.concat([csv, new_row], ignore_index=True)

    csv.loc[csv["Model"] == model, "Index vel. R2 Score"] = float(score1)
    csv.loc[csv["Model"] == model, "MRS vel. R2 Score"] = float(score2)
    csv.to_csv(get_project_root() / "results" / "r2_scores.csv", index=False)