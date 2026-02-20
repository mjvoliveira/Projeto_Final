import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


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


def get_corr_features() -> list[str]:
    # feature chave = pred_center (a previsão do modelo por-centro)
    return [
        "center_id",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "week_sin", "week_cos",
        "month_sin", "month_cos",
        "is_month_end",
        "is_end_of_month_window",
        "pred_center",
    ]


def fit_global_residual_corrector(train_df: pd.DataFrame) -> HistGradientBoostingRegressor:
    """
    Aprende residual = y_true - pred_center.
    Treina global, usando center_id + calendário + pred_center.
    """
    d = train_df.copy()
    d["residual"] = d["y_true"].to_numpy(dtype=float) - d["pred_center"].to_numpy(dtype=float)

    X = d[get_corr_features()]
    y = d["residual"].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=700,
        min_samples_leaf=120,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def apply_global_residual_corrector(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> np.ndarray:
    X = df[get_corr_features()]
    corr = model.predict(X)
    return np.clip(df["pred_center"].to_numpy(dtype=float) + corr, 0, None)