import pandas as pd

from src.models_hourly.io import load_hourly_core
from src.models_hourly.models import (
    mae, rmse, wape,
    add_time_features,
    compute_operational_H,
    naive_last,
    seasonal_naive,
    mean_by_hour_dow_forecast,
)

DAYS_PER_WEEK = 6
MIN_TRAIN_DAYS = 30
STEP_DAYS = 1
HORIZON_WEEKS = [1, 8]  # 1 semana e 8 semanas


def rolling_baselines_center(dfc: pd.DataFrame) -> pd.DataFrame:
    dfc = dfc.copy().sort_values("hora").reset_index(drop=True)

    H = compute_operational_H(dfc, time_col="hora")
    if H <= 0:
        return pd.DataFrame()

    min_train_size = MIN_TRAIN_DAYS * H
    step = STEP_DAYS * H

    seasonal_daily = H
    seasonal_weekly = DAYS_PER_WEEK * H
    horizons = [w * DAYS_PER_WEEK * H for w in HORIZON_WEEKS]

    y_full = dfc["y_inspecoes"].reset_index(drop=True)

    rows = []
    for h in horizons:
        if len(dfc) <= min_train_size + h:
            continue

        for t in range(min_train_size, len(dfc) - h, step):
            train_raw = dfc.iloc[:t].copy()
            test_raw  = dfc.iloc[t:t+h].copy()

            y_train = train_raw["y_inspecoes"]
            y_test = test_raw["y_inspecoes"].to_numpy()

            p = naive_last(y_train, h)
            rows.append({"h": h, "model": "naive_last", "mae": mae(y_test, p), "rmse": rmse(y_test, p), "wape": wape(y_test, p)})

            if t >= seasonal_daily:
                p = seasonal_naive(y_full, t, h, seasonal_daily)
                rows.append({"h": h, "model": f"seasonal_naive_{seasonal_daily}", "mae": mae(y_test, p), "rmse": rmse(y_test, p), "wape": wape(y_test, p)})

            if t >= seasonal_weekly:
                p = seasonal_naive(y_full, t, h, seasonal_weekly)
                rows.append({"h": h, "model": f"seasonal_naive_{seasonal_weekly}", "mae": mae(y_test, p), "rmse": rmse(y_test, p), "wape": wape(y_test, p)})

            train_tf = add_time_features(train_raw, time_col="hora")
            test_tf  = add_time_features(test_raw, time_col="hora")
            p = mean_by_hour_dow_forecast(train_tf, test_tf, target_col="y_inspecoes")
            rows.append({"h": h, "model": "mean_by_hour_dow", "mae": mae(y_test, p), "rmse": rmse(y_test, p), "wape": wape(y_test, p)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_hourly_core()

    all_rows = []
    for cid, g in df.groupby("CFGCENTROID"):
        g = g.sort_values("hora")
        res = rolling_baselines_center(g)
        if len(res) == 0:
            continue
        res.insert(0, "CFGCENTROID", cid)
        all_rows.append(res)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary = (
        out.groupby(["model", "h"], as_index=False)[["mae","rmse","wape"]]
           .mean()
           .sort_values(["h","mae"])
    )

    print("\nBaseline results (rolling-origin evaluation):\n")
    print(summary.round(4))