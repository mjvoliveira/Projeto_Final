import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# Metrics
# ============================================================
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def mape_pos(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


# ============================================================
# Operational H
# ============================================================
def compute_operational_H(df_center: pd.DataFrame, time_col: str = "hora") -> int:
    dfc = df_center.copy()
    dfc[time_col] = pd.to_datetime(dfc[time_col], errors="coerce")
    dfc = dfc.dropna(subset=[time_col])
    if dfc.empty:
        return 0

    dfc["date"] = dfc[time_col].dt.date
    dfc["hour"] = dfc[time_col].dt.hour
    counts = dfc.groupby("date")["hour"].nunique()
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0
    return max(int(np.median(counts.to_numpy())), 0)


# ============================================================
# Time features
# ============================================================
def add_time_features(df: pd.DataFrame, time_col: str = "hora") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])

    out["hour_of_day"] = out[time_col].dt.hour
    out["dow"] = out[time_col].dt.dayofweek
    out["month"] = out[time_col].dt.month
    out["week_of_year"] = out[time_col].dt.isocalendar().week.astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24.0)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52.0)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    out["is_month_end"] = out[time_col].dt.is_month_end.astype(int)
    day = out[time_col].dt.day
    dim = out[time_col].dt.daysinmonth
    out["is_end_of_month_window"] = ((dim - day) <= 2).astype(int)

    return out


# ============================================================
# Supervised build (timestamp merges) + rollmean
# ============================================================
def build_supervised(
    df_center: pd.DataFrame,
    target_col: str = "y_inspecoes",
    time_col: str = "hora",
) -> pd.DataFrame:
    df = df_center.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    df = add_time_features(df, time_col=time_col)

    base = df[[time_col, target_col]].copy()

    df["__t_lag_1h"] = df[time_col] - pd.Timedelta(hours=1)
    df = df.merge(
        base.rename(columns={time_col: "__t_lag_1h", target_col: "lag_1h"}),
        on="__t_lag_1h",
        how="left",
    )

    df["__t_lag_1d"] = df[time_col] - pd.Timedelta(days=1)
    df = df.merge(
        base.rename(columns={time_col: "__t_lag_1d", target_col: "lag_1d"}),
        on="__t_lag_1d",
        how="left",
    )

    df["__t_lag_7d"] = df[time_col] - pd.Timedelta(days=7)
    df = df.merge(
        base.rename(columns={time_col: "__t_lag_7d", target_col: "lag_7d"}),
        on="__t_lag_7d",
        how="left",
    )

    tmp = df[[time_col, target_col]].set_index(time_col).sort_index()
    df = df.merge(
        tmp[target_col].rolling("24h").mean().shift(1).rename("rollmean_24h").reset_index(),
        on=time_col,
        how="left",
    )

    df = df.drop(columns=[c for c in df.columns if c.startswith("__t_lag_")])
    return df


def get_feature_cols(_: pd.DataFrame | None = None) -> list[str]:
    return [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "week_sin", "week_cos",
        "month_sin", "month_cos",
        "is_month_end",
        "is_end_of_month_window",
        "lag_1h", "lag_1d", "lag_7d",
        "rollmean_24h",
    ]


# ============================================================
# Baselines
# ============================================================
def naive_last(y_train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(float(y_train.iloc[-1]), h)


def seasonal_naive(y_full: pd.Series, split_idx: int, h: int, season: int) -> np.ndarray:
    preds = []
    for i in range(h):
        j = split_idx + i - season
        if j < 0:
            j = 0
        preds.append(float(y_full.iloc[j]))
    return np.asarray(preds, dtype=float)


def mean_by_hour_dow_forecast(train_tf: pd.DataFrame, test_tf: pd.DataFrame, target_col: str = "y_inspecoes") -> np.ndarray:
    g = train_tf.groupby(["dow", "hour_of_day"])[target_col].mean()
    keys = list(zip(test_tf["dow"].to_numpy(), test_tf["hour_of_day"].to_numpy()))
    out = np.array([float(g.get(k, train_tf[target_col].mean())) for k in keys], dtype=float)
    return np.clip(out, 0, None)


# ============================================================
# Models pack
# ============================================================
@dataclass
class HourlyModelPack:
    ridge: Ridge
    hgb: HistGradientBoostingRegressor


def build_models(random_state: int = 42, hgb_params: dict | None = None) -> HourlyModelPack:
    ridge = Ridge(alpha=1.0, random_state=random_state)

    base_params = dict(
        loss="absolute_error",
        learning_rate=0.03,
        max_depth=5,
        max_iter=900,
        min_samples_leaf=30,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    if hgb_params:
        base_params.update(hgb_params)

    hgb = HistGradientBoostingRegressor(**base_params)
    return HourlyModelPack(ridge=ridge, hgb=hgb)


# ============================================================
# Helper: pick ensemble weight on validation
# ============================================================
def pick_best_w_wape(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, grid=None) -> float:
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)  # step 0.05
    best_w = 1.0
    best = float("inf")
    for w in grid:
        p = w * p1 + (1.0 - w) * p2
        val = wape(y_true, p)
        if val < best:
            best = val
            best_w = float(w)
    return best_w


# ============================================================
# Fit/predict per centro (API do backtest)
# - inclui: calibração multiplicativa + ensemble com w otimizado em validação
# ============================================================
def fit_predict_models(
    df_center: pd.DataFrame,
    test_h: int,
    target_col: str = "y_inspecoes",
    time_col: str = "hora",
    days_per_week: int = 6,
    random_state: int = 42,
    hgb_params: dict | None = None,
) -> dict | None:
    dfc = df_center.copy()
    dfc[time_col] = pd.to_datetime(dfc[time_col], errors="coerce")
    dfc = dfc.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if len(dfc) <= test_h + 30:
        return None

    H = compute_operational_H(dfc, time_col=time_col)
    if H <= 0:
        return None

    split_idx = len(dfc) - test_h
    train_raw = dfc.iloc[:split_idx].copy()
    test_raw = dfc.iloc[split_idx:].copy()

    y_full = dfc[target_col].reset_index(drop=True)
    y_train = train_raw[target_col]
    y_test_raw = test_raw[target_col].to_numpy()

    preds: dict[str, object] = {}

    # ---- baselines
    preds["naive_last"] = naive_last(y_train, test_h)

    if split_idx >= H:
        preds[f"seasonal_naive_{H}"] = seasonal_naive(y_full, split_idx, test_h, H)

    if split_idx >= days_per_week * H:
        preds[f"seasonal_naive_{days_per_week * H}"] = seasonal_naive(
            y_full, split_idx, test_h, days_per_week * H
        )

    train_tf = add_time_features(train_raw, time_col=time_col)
    test_tf = add_time_features(test_raw, time_col=time_col)
    preds["mean_by_hour_dow"] = mean_by_hour_dow_forecast(train_tf, test_tf, target_col=target_col)

    # ---- supervised ML
    sup = build_supervised(dfc, target_col=target_col, time_col=time_col).sort_values(time_col).reset_index(drop=True)

    test_start = test_raw[time_col].min()
    train_sup_all = sup[sup[time_col] < test_start].copy()
    test_sup = sup[sup[time_col] >= test_start].copy().tail(test_h)

    feat_cols = get_feature_cols(sup)

    # validação temporal: último 1 semana operacional do treino (aprox)
    val_h = min(test_h, max(2 * H * days_per_week, 30))  # ~1-2 semanas ou mínimo 30
    if len(train_sup_all) <= val_h + 100:
        val_h = max(30, min(val_h, len(train_sup_all) // 5))

    train_sup = train_sup_all.iloc[:-val_h].copy()
    val_sup = train_sup_all.iloc[-val_h:].copy()

    # imputação pelo treino-fit (sem leakage)
    med = train_sup[feat_cols].median(numeric_only=True)
    train_sup[feat_cols] = train_sup[feat_cols].fillna(med)
    val_sup[feat_cols] = val_sup[feat_cols].fillna(med)
    test_sup[feat_cols] = test_sup[feat_cols].fillna(med)

    train_sup = train_sup.dropna(subset=[target_col])
    val_sup = val_sup.dropna(subset=[target_col])
    test_sup = test_sup.dropna(subset=[target_col]).tail(test_h)

    if len(train_sup) >= 200 and len(val_sup) >= 30 and len(test_sup) >= max(20, test_h // 4):
        X_train = train_sup[feat_cols]
        y_train_sup = np.log1p(train_sup[target_col].to_numpy(dtype=float))

        X_val = val_sup[feat_cols]
        y_val = val_sup[target_col].to_numpy(dtype=float)

        X_test = test_sup[feat_cols]
        y_true_test = test_sup[target_col].to_numpy(dtype=float)

        pack = build_models(random_state=random_state, hgb_params=hgb_params)

        # ---- Ridge
        pack.ridge.fit(X_train, y_train_sup)
        p_val_r = np.expm1(pack.ridge.predict(X_val))
        p_test_r = np.expm1(pack.ridge.predict(X_test))
        p_val_r = np.clip(p_val_r, 0, None)
        p_test_r = np.clip(p_test_r, 0, None)
        preds["ridge"] = (p_test_r, y_true_test)

        # ---- HGB
        pack.hgb.fit(X_train, y_train_sup)
        p_val_h = np.expm1(pack.hgb.predict(X_val))
        p_test_h = np.expm1(pack.hgb.predict(X_test))
        p_val_h = np.clip(p_val_h, 0, None)
        p_test_h = np.clip(p_test_h, 0, None)

        # ---- calibração multiplicativa (remove bias sistemático)
        # scale = sum(y_val) / sum(p_val_h)
        scale = (np.sum(y_val) + 1e-8) / (np.sum(p_val_h) + 1e-8)
        scale = float(np.clip(scale, 0.6, 1.4))  # evita overfit extremo
        p_val_h_cal = p_val_h * scale
        p_test_h_cal = p_test_h * scale

        preds["hgb"] = (np.clip(p_test_h_cal, 0, None), y_true_test)

        # ---- mean_by_hour_dow alinhado ao test_sup (para ensemble)
        test_aligned_tf = add_time_features(test_sup[[time_col]].copy(), time_col=time_col)
        mean_test_aligned = mean_by_hour_dow_forecast(train_tf, test_aligned_tf, target_col=target_col)

        # e alinhado à val_sup (para escolher w sem leakage)
        val_aligned_tf = add_time_features(val_sup[[time_col]].copy(), time_col=time_col)
        mean_val_aligned = mean_by_hour_dow_forecast(train_tf, val_aligned_tf, target_col=target_col)

        # ---- escolher w na validação (WAPE)
        best_w = pick_best_w_wape(y_val, p_val_h_cal, mean_val_aligned)

        ens_test = best_w * p_test_h_cal + (1.0 - best_w) * mean_test_aligned
        ens_test = np.clip(ens_test, 0, None)
        preds["ens_hgb_mean"] = (ens_test, y_true_test)

    # ---- metrics + sums
    MAE, RMSE, WAPE, MAPE, MAPE_POS = {}, {}, {}, {}, {}
    SUMY, SUMABSERR, SUMABSERR_POS, NPOS = {}, {}, {}, {}

    for name, p in preds.items():
        if isinstance(p, tuple):
            y_pred, y_true = p
        else:
            y_pred, y_true = np.asarray(p, dtype=float), y_test_raw

        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        MAE[name] = mae(y_true, y_pred)
        RMSE[name] = rmse(y_true, y_pred)
        WAPE[name] = wape(y_true, y_pred)
        MAPE[name] = mape(y_true, y_pred)
        MAPE_POS[name] = mape_pos(y_true, y_pred)

        abs_err = np.abs(y_true - y_pred)
        SUMY[name] = float(np.sum(y_true))
        SUMABSERR[name] = float(np.sum(abs_err))

        mask_pos = y_true > 0
        SUMABSERR_POS[name] = float(np.sum(abs_err[mask_pos])) if np.any(mask_pos) else 0.0
        NPOS[name] = int(np.sum(mask_pos))

    return {
        "H": H,
        "MAE": MAE,
        "RMSE": RMSE,
        "WAPE": WAPE,
        "MAPE": MAPE,
        "MAPE_POS": MAPE_POS,
        "SUMY": SUMY,
        "SUMABSERR": SUMABSERR,
        "SUMABSERR_POS": SUMABSERR_POS,
        "NPOS": NPOS,
    }


# ============================================================
# predict_test_window (para plots)
# ============================================================
def predict_test_window(
    df_center: pd.DataFrame,
    test_h: int,
    model_name: str = "hgb",
    target_col: str = "y_inspecoes",
    time_col: str = "hora",
    days_per_week: int = 6,
    random_state: int = 42,
) -> pd.DataFrame | None:
    dfc = df_center.copy()
    dfc[time_col] = pd.to_datetime(dfc[time_col], errors="coerce")
    dfc = dfc.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if len(dfc) <= test_h + 30:
        return None

    H = compute_operational_H(dfc, time_col=time_col)
    if H <= 0:
        return None

    split_idx = len(dfc) - test_h
    train_raw = dfc.iloc[:split_idx].copy()
    test_raw = dfc.iloc[split_idx:].copy()
    test_start = test_raw[time_col].min()

    sup = build_supervised(dfc, target_col=target_col, time_col=time_col).sort_values(time_col).reset_index(drop=True)
    train_sup_all = sup[sup[time_col] < test_start].copy()
    test_sup = sup[sup[time_col] >= test_start].copy().tail(test_h)

    feat_cols = get_feature_cols(sup)

    val_h = min(test_h, max(2 * H * days_per_week, 30))
    train_sup = train_sup_all.iloc[:-val_h].copy()
    val_sup = train_sup_all.iloc[-val_h:].copy()

    med = train_sup[feat_cols].median(numeric_only=True)
    train_sup[feat_cols] = train_sup[feat_cols].fillna(med)
    val_sup[feat_cols] = val_sup[feat_cols].fillna(med)
    test_sup[feat_cols] = test_sup[feat_cols].fillna(med)

    train_sup = train_sup.dropna(subset=[target_col])
    val_sup = val_sup.dropna(subset=[target_col])
    test_sup = test_sup.dropna(subset=[target_col]).tail(test_h)
    if len(train_sup) < 200 or len(val_sup) < 30 or len(test_sup) < 20:
        return None

    X_train = train_sup[feat_cols]
    y_train_sup = np.log1p(train_sup[target_col].to_numpy(dtype=float))
    X_val = val_sup[feat_cols]
    y_val = val_sup[target_col].to_numpy(dtype=float)
    X_test = test_sup[feat_cols]
    y_true = test_sup[target_col].to_numpy(dtype=float)

    pack = build_models(random_state=random_state)

    train_tf = add_time_features(train_raw, time_col=time_col)

    if model_name == "mean_by_hour_dow":
        test_aligned_tf = add_time_features(test_sup[[time_col]].copy(), time_col=time_col)
        y_pred = mean_by_hour_dow_forecast(train_tf, test_aligned_tf, target_col=target_col)
        y_pred = np.clip(y_pred, 0, None)

    else:
        if model_name == "ridge":
            model = pack.ridge
        else:
            model = pack.hgb

        model.fit(X_train, y_train_sup)

        p_val = np.expm1(model.predict(X_val))
        p_test = np.expm1(model.predict(X_test))
        p_val = np.clip(p_val, 0, None)
        p_test = np.clip(p_test, 0, None)

        # calibração multiplicativa para hgb e ridge (consistente com fit_predict_models)
        scale = (np.sum(y_val) + 1e-8) / (np.sum(p_val) + 1e-8)
        scale = float(np.clip(scale, 0.6, 1.4))
        p_test = np.clip(p_test * scale, 0, None)

        if model_name == "ens_hgb_mean":
            # construir mean aligned
            test_aligned_tf = add_time_features(test_sup[[time_col]].copy(), time_col=time_col)
            mean_test = mean_by_hour_dow_forecast(train_tf, test_aligned_tf, target_col=target_col)
            val_aligned_tf = add_time_features(val_sup[[time_col]].copy(), time_col=time_col)
            mean_val = mean_by_hour_dow_forecast(train_tf, val_aligned_tf, target_col=target_col)

            p_val_cal = np.clip(p_val * scale, 0, None)
            best_w = pick_best_w_wape(y_val, p_val_cal, mean_val)
            p_test = np.clip(best_w * p_test + (1.0 - best_w) * mean_test, 0, None)

        y_pred = p_test

    out = test_sup[[time_col]].copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    return out