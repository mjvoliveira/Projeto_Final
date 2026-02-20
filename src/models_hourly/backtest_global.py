import numpy as np
import pandas as pd

from src.models_hourly.io import load_hourly_core
from src.models_hourly.models_global import global_backtest_predictions
from src.models_hourly.models import mae, rmse, wape, mape, mape_pos  # reutiliza métricas


DAYS_PER_WEEK = 6
TEST_WEEKS_SHORT = 1
TEST_WEEKS_LONG = 8


def summarize(pred_frame: pd.DataFrame, model_col: str):
    rows = []
    for cid, g in pred_frame.groupby("CFGCENTROID"):
        y = g["y_true"].to_numpy()
        p = g[model_col].to_numpy()
        rows.append(
            dict(
                CFGCENTROID=cid,
                mae=mae(y, p),
                rmse=rmse(y, p),
                wape=wape(y, p),
                mape=mape(y, p),
                mape_pos=mape_pos(y, p),
                test_sum_y=float(np.sum(y)),
                test_sum_abs_err=float(np.sum(np.abs(y - p))),
            )
        )
    dfc = pd.DataFrame(rows)

    wape_global = dfc["test_sum_abs_err"].sum() / (dfc["test_sum_y"].sum() + 1e-8)
    simple = dfc[["mae", "rmse", "wape", "mape", "mape_pos"]].mean().to_dict()
    return dfc, wape_global, simple


def main():
    df = load_hourly_core()
    if df.empty:
        print("Dataset vazio.")
        return

    # estimar H a partir do centro com mais dados
    top_center = df["CFGCENTROID"].value_counts().index[0]
    H = df[df["CFGCENTROID"] == top_center]["hora"].dt.hour.nunique()
    if H <= 0:
        H = 12

    TEST_H_SHORT = TEST_WEEKS_SHORT * DAYS_PER_WEEK * H
    TEST_H_LONG = TEST_WEEKS_LONG * DAYS_PER_WEEK * H

    for horizon, test_h in [("short_1week", TEST_H_SHORT), ("long_8weeks", TEST_H_LONG)]:
        res = global_backtest_predictions(df, test_h=test_h)
        pred = res["pred_frame"]

        for name, col in [
            ("hgb_global", "pred_hgb"),
            ("ridge_global", "pred_ridge"),
            ("mean_by_hour_dow", "pred_mean_by_hour_dow"),
        ]:
            per_center, wape_g, simple = summarize(pred, col)
            print(f"\n[{horizon}] {name} | WAPE_global={wape_g:.4f} | mae_mean={simple['mae']:.4f} | rmse_mean={simple['rmse']:.4f}")

            # top contrib
            per_center["err_contrib"] = per_center["test_sum_abs_err"] / (per_center["test_sum_abs_err"].sum() + 1e-8)
            top = per_center.sort_values("err_contrib", ascending=False).head(10)[
                ["CFGCENTROID", "err_contrib", "wape", "test_sum_y", "test_sum_abs_err"]
            ]
            print("Top 10 contribuintes para erro global:")
            print(top.round(4).to_string(index=False))


if __name__ == "__main__":
    main()