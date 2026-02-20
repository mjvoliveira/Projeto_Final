import numpy as np
import pandas as pd
from pathlib import Path

from src.models_hourly.io import load_hourly_core
from src.models_hourly.models import (
    build_supervised,
    compute_operational_H,
    build_models,
    get_feature_cols,
    make_future_features,
)

DAYS_PER_WEEK = 6
FORECAST_WEEKS = 8
OUT_PATH = Path("/Users/catarinatorres/Documents/ControlAuto/Data/Raw/controlauto_outputs/forecast_hourly_all_centers_hgb_v1.parquet")


def build_future_index(df_center: pd.DataFrame, steps: int) -> pd.DatetimeIndex:
    g = df_center.copy()
    g["hora"] = pd.to_datetime(g["hora"])
    last_ts = g["hora"].max()

    op_hours = sorted(g["hora"].dt.hour.unique().tolist())
    future = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=steps * 4, freq="h")
    future = future[future.hour.isin(op_hours)]
    return future[:steps]


def seasonal_daily_fallback(y_hist: list[float], H: int, steps: int) -> np.ndarray:
    if len(y_hist) >= H:
        pattern = np.array(y_hist[-H:], dtype=float)
        return np.resize(pattern, steps)
    last = y_hist[-1] if len(y_hist) else 0.0
    return np.repeat(last, steps)


if __name__ == "__main__":
    df = load_hourly_core()

    results = []
    for cid, g in df.groupby("CFGCENTROID"):
        g = g.sort_values("hora").reset_index(drop=True)
        H = compute_operational_H(g, time_col="hora")
        if H <= 0:
            continue

        steps = FORECAST_WEEKS * DAYS_PER_WEEK * H
        future_idx = build_future_index(g, steps=steps)
        if len(future_idx) == 0:
            continue

        # ---- Treino supervised ----
        try:
            sup = build_supervised(g, target_col="y_inspecoes", H=H, days_per_week=DAYS_PER_WEEK)
            sup = sup.sort_values("hora").reset_index(drop=True)

            feat_cols = get_feature_cols(sup)
            X_train = sup[feat_cols]
            y_train = sup["y_inspecoes"]

            model_pack = build_models()
            model = model_pack.hgb
            model.fit(X_train, y_train)

            # ---- Forecast recursivo ----
            y_hist = g["y_inspecoes"].astype(float).tolist()
            preds = []

            for ts in future_idx:
                feats = make_future_features(ts, y_hist=y_hist, H=H, days_per_week=DAYS_PER_WEEK)
                X_row = pd.DataFrame([feats])[feat_cols]
                y_hat = float(model.predict(X_row)[0])
                y_hat = max(0.0, y_hat)           # counts não negativos
                preds.append(y_hat)
                y_hist.append(y_hat)

            y_hat = np.array(preds, dtype=float)

        except Exception as e:
            # fallback robusto
            y_hist = g["y_inspecoes"].astype(float).tolist()
            y_hat = seasonal_daily_fallback(y_hist, H=H, steps=len(future_idx))

        out = pd.DataFrame({
            "CFGCENTROID": cid,
            "hora": future_idx,
            "y_hat": np.round(y_hat).astype(float),
            "model": "hgb",
            "H": H,
        })
        results.append(out)

    forecast_all = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    forecast_all.to_parquet(OUT_PATH, index=False)

    print("Forecast (HGB) criado para", forecast_all["CFGCENTROID"].nunique() if len(forecast_all) else 0, "centros")
    print("Guardado:", OUT_PATH)