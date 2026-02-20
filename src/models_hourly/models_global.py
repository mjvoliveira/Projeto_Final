import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# Metrics
# ============================================================
def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


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


def mean_by_hour_dow_forecast(train_tf: pd.DataFrame, test_tf: pd.DataFrame, target_col: str) -> np.ndarray:
    g = train_tf.groupby(["dow", "hour_of_day"])[target_col].mean()
    keys = list(zip(test_tf["dow"].to_numpy(), test_tf["hour_of_day"].to_numpy()))
    out = np.array([float(g.get(k, train_tf[target_col].mean())) for k in keys], dtype=float)
    return np.clip(out, 0, None)


# ============================================================
# Global supervised build
# ============================================================
def build_supervised_global(
    df: pd.DataFrame,
    center_col: str = "CFGCENTROID",
    target_col: str = "y_inspecoes",
    time_col: str = "hora",
) -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, center_col]).sort_values([center_col, time_col]).reset_index(drop=True)

    d["center_id"] = d[center_col].astype("category").cat.codes.astype(int)
    d = add_time_features(d, time_col=time_col)

    base = d[[center_col, time_col, target_col]].copy()

    d["__t_lag_1h"] = d[time_col] - pd.Timedelta(hours=1)
    d = d.merge(
        base.rename(columns={time_col: "__t_lag_1h", target_col: "lag_1h"}),
        on=[center_col, "__t_lag_1h"],
        how="left",
    )

    d["__t_lag_1d"] = d[time_col] - pd.Timedelta(days=1)
    d = d.merge(
        base.rename(columns={time_col: "__t_lag_1d", target_col: "lag_1d"}),
        on=[center_col, "__t_lag_1d"],
        how="left",
    )

    d["__t_lag_7d"] = d[time_col] - pd.Timedelta(days=7)
    d = d.merge(
        base.rename(columns={time_col: "__t_lag_7d", target_col: "lag_7d"}),
        on=[center_col, "__t_lag_7d"],
        how="left",
    )

    # rolling 24h real por centro (sem leakage)
    roll_parts = []
    for cid, g in d[[center_col, time_col, target_col]].groupby(center_col, sort=False):
        gg = g.sort_values(time_col).set_index(time_col)
        r = gg[target_col].rolling("24h").mean().shift(1).rename("rollmean_24h").reset_index()
        r[center_col] = cid
        roll_parts.append(r)

    roll = pd.concat(roll_parts, ignore_index=True)
    d = d.merge(roll, on=[center_col, time_col], how="left")

    d = d.drop(columns=[c for c in d.columns if c.startswith("__t_lag_")])
    return d


def get_feature_cols_global() -> list[str]:
    return [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "week_sin", "week_cos",
        "month_sin", "month_cos",
        "is_month_end",
        "is_end_of_month_window",
        "center_id",
        "lag_1h", "lag_1d", "lag_7d",
        "rollmean_24h",
    ]


# ============================================================
# Models
# ============================================================
@dataclass
class GlobalModelPack:
    ridge: Ridge
    hgb: HistGradientBoostingRegressor


def build_models(random_state: int = 42, hgb_params: dict | None = None) -> GlobalModelPack:
    ridge = Ridge(alpha=1.0, random_state=random_state)

    base_params = dict(
        loss="absolute_error",
        learning_rate=0.03,
        max_depth=6,
        max_iter=1200,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    if hgb_params:
        base_params.update(hgb_params)

    hgb = HistGradientBoostingRegressor(**base_params)
    return GlobalModelPack(ridge=ridge, hgb=hgb)


# ============================================================
# Global backtest + calibração por centro
# ============================================================
def global_backtest_predictions(
    df: pd.DataFrame,
    test_h: int,
    center_col: str = "CFGCENTROID",
    target_col: str = "y_inspecoes",
    time_col: str = "hora",
    random_state: int = 42,
    hgb_params: dict | None = None,
    use_log1p: bool = True,
    val_h: int | None = None,
    scale_clip: tuple[float, float] = (0.6, 1.4),
) -> dict:
    """
    1) Split por centro: últimas test_h = teste
    2) Val por centro: janela imediatamente antes do teste (val_h)
    3) Treino global em (train)
    4) Pred val/test
    5) Calcula scale por centro via val e aplica ao test (HGB)
    """

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, center_col]).sort_values([center_col, time_col]).reset_index(drop=True)

    sup = build_supervised_global(d, center_col=center_col, target_col=target_col, time_col=time_col)
    feat_cols = get_feature_cols_global()

    # ranking reverse por centro
    sup["__rank_rev"] = sup.groupby(center_col).cumcount(ascending=False)

    # val_h default: 1 semana (em horas) ~ test_h/8, mas mínimo 30, máximo test_h
    if val_h is None:
        val_h = max(30, min(test_h, int(round(test_h / 8))))

    sup["__is_test"] = sup["__rank_rev"] < test_h
    sup["__is_val"] = (sup["__rank_rev"] >= test_h) & (sup["__rank_rev"] < test_h + val_h)

    train_sup = sup[~sup["__is_test"] & ~sup["__is_val"]].copy()
    val_sup = sup[sup["__is_val"]].copy()
    test_sup = sup[sup["__is_test"]].copy()

    # imputação global (treino)
    med = train_sup[feat_cols].median(numeric_only=True)
    for part in (train_sup, val_sup, test_sup):
        part[feat_cols] = part[feat_cols].fillna(med)

    train_sup = train_sup.dropna(subset=[target_col])
    val_sup = val_sup.dropna(subset=[target_col])
    test_sup = test_sup.dropna(subset=[target_col])

    X_train = train_sup[feat_cols]
    y_train = train_sup[target_col].to_numpy(dtype=float)

    X_val = val_sup[feat_cols]
    y_val = val_sup[target_col].to_numpy(dtype=float)

    X_test = test_sup[feat_cols]
    y_test = test_sup[target_col].to_numpy(dtype=float)

    y_train_fit = np.log1p(y_train) if use_log1p else y_train

    pack = build_models(random_state=random_state, hgb_params=hgb_params)
    pack.hgb.fit(X_train, y_train_fit)
    pack.ridge.fit(X_train, y_train_fit)

    # preds
    p_val_h = pack.hgb.predict(X_val)
    p_test_h = pack.hgb.predict(X_test)

    p_val_r = pack.ridge.predict(X_val)
    p_test_r = pack.ridge.predict(X_test)

    if use_log1p:
        p_val_h = np.expm1(p_val_h)
        p_test_h = np.expm1(p_test_h)
        p_val_r = np.expm1(p_val_r)
        p_test_r = np.expm1(p_test_r)

    p_val_h = np.clip(p_val_h, 0, None)
    p_test_h = np.clip(p_test_h, 0, None)
    p_val_r = np.clip(p_val_r, 0, None)
    p_test_r = np.clip(p_test_r, 0, None)

    # calibração por centro via validação (aplicada ao TESTE no HGB)
    val_df = val_sup[[center_col]].copy()
    val_df["y_val"] = y_val
    val_df["p_val_h"] = p_val_h

    scales = {}
    for cid, g in val_df.groupby(center_col):
        s = (g["y_val"].sum() + 1e-8) / (g["p_val_h"].sum() + 1e-8)
        s = float(np.clip(s, scale_clip[0], scale_clip[1]))
        scales[cid] = s

    # aplicar scales no teste
    test_cids = test_sup[center_col].to_numpy()
    scale_vec = np.array([scales.get(cid, 1.0) for cid in test_cids], dtype=float)
    p_test_h_cal = np.clip(p_test_h * scale_vec, 0, None)

    # baseline mean_by_hour_dow por centro (alinhado por index)
    mean_series = pd.Series(index=test_sup.index, dtype=float)
    for cid, g in test_sup.groupby(center_col, sort=False):
        train_raw_c = d[d[center_col] == cid].sort_values(time_col)
        # remove val+test do fim
        cut = test_h + val_h
        if len(train_raw_c) > cut:
            train_raw_c = train_raw_c.iloc[:-cut].copy()
        train_tf = add_time_features(train_raw_c, time_col=time_col)

        test_tf = add_time_features(g[[time_col]].copy(), time_col=time_col)
        mp = mean_by_hour_dow_forecast(train_tf, test_tf, target_col=target_col)
        mean_series.loc[g.index] = mp

    pm = np.clip(mean_series.to_numpy(dtype=float), 0, None)

    out = test_sup[[center_col, time_col]].copy()
    out["y_true"] = y_test
    out["pred_hgb"] = p_test_h_cal
    out["pred_ridge"] = p_test_r
    out["pred_mean_by_hour_dow"] = pm

    return {"pred_frame": out, "feat_cols": feat_cols, "val_h": val_h}