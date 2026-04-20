import pandas as pd
from project_brain_decoder.config import get_project_root

def get_scores(model, score1, score2):
    try:
        csv = pd.read_csv(get_project_root() / "results" / "r2_scores.csv")
    except pd.errors.EmptyDataError:
        csv = pd.DataFrame(columns=["Model", "Index vel. R2 Score", "MRS vel. R2 Score"])
    if model not in csv["Model"].values:
        r2_dict = {"Model": [model], "Index vel. R2 Score": [score1], "MRS vel. R2 Score": [score2]}
        df = pd.DataFrame(r2_dict)
        csv = pd.concat([csv, df])
    csv.loc[csv["Model"] == model, "Index vel. R2 Score"] = float(score1)
    csv.loc[csv["Model"] == model, "MRS vel. R2 Score"] = float(score2)
    csv.to_csv(get_project_root() / "results" / "r2_scores.csv", index=False)