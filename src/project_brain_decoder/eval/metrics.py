import pandas as pd


results = dict()
results["Ridge"] = get_r2(y_true_ridge, y_pred_ridge)
results["TCN"] = get_r2(y_true_tcn, y_pred_tcn)
results["Kalman"] = get_r2(y_true_kalman, y_pred_kalman)
results["GRU"] = get_r2(y_true_gru, y_pred_gru)
results["Transformer"] = get_r2(y_true_transf, y_pred_transf)

pd.DataFrame.from_dict(results, orient="index").to_csv("results/r2_comparison.csv")

