import numpy as np
import pandas as pd

from src.models_hourly.io import load_hourly_core
from src.models_hourly.models import (
    compute_operational_H,
    build_supervised,
    get_feature_cols,
    build_models,
    mae,
    rmse,
    wape,
    mape,
    mape_pos,
)
from src.models_hourly.models_residual_global import add_time_features, fit_global_residual_corrector, apply_global_residual_corrector

DAYS_PER_WEEK = 6
TEST_WEEKS_SHORT = 1
TEST_WEEKS_LONG = 8
MIN_EXTRA_TRAIN_DAYS = 30


def estimate_H_global(df: pd.DataFrame) -> int:
    top_center = df["CFGCENTROID"].value_counts().index[0]
    g = df[df["CFGCENTROID"] == top_center].copy()
    return compute_operational_H(g, time_col="hora")


def build_weighted_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    g = results_df.groupby(["horizon", "model"], as_index=False)
    out = g.agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        wape_mean=("wape", "mean"),
        mape_mean=("mape", "mean"),
        mape_pos_mean=("mape_pos", "mean"),
        sum_y=("test_sum_y", "sum"),
        sum_abs_err=("test_sum_abs_err", "sum"),
        n_centers=("CFGCENTROID", "nunique"),
    )
    out["WAPE_global"] = out["sum_abs_err"] / (out["sum_y"] + 1e-8)
    out = out.sort_values(["horizon", "WAPE_global"])
    return out


def _split_train_val_test_by_time(sup: pd.DataFrame, test_h: int, val_h: int, time_col: str = "hora"):
    sup = sup.sort_values(time_col).reset_index(drop=True)
    if len(sup) <= test_h + val_h + 200:
        return None, None, None

    test_start = sup[time_col].iloc[-test_h]
    val_start = sup[time_col].iloc[-(test_h + val_h)]

    train = sup[sup[time_col] < val_start].copy()
    val = sup[(sup[time_col] >= val_start) & (sup[time_col] < test_start)].copy()
    test = sup[sup[time_col] >= test_start].copy()

    return train, val, test


def _train_center_hgb_and_predict(sup_center: pd.DataFrame, test_h: int, val_h: int):
    feat_cols = get_feature_cols(sup_center)
    train, val, test = _split_train_val_test_by_time(sup_center, test_h=test_h, val_h=val_h)
    if train is None:
        return None

    # impute pelo treino (sem leakage)
    med = train[feat_cols].median(numeric_only=True)
    for part in (train, val, test):
        part[feat_cols] = part[feat_cols].fillna(med)

    train = train.dropna(subset=["y_inspecoes"])
    val = val.dropna(subset=["y_inspecoes"])
    test = test.dropna(subset=["y_inspecoes"])

    if len(train) < 250 or len(val) < 30 or len(test) < 30:
        return None

    X_train = train[feat_cols]
    y_train = np.log1p(train["y_inspecoes"].to_numpy(dtype=float))

    X_val = val[feat_cols]
    y_val = val["y_inspecoes"].to_numpy(dtype=float)

    X_test = test[feat_cols]
    y_test = test["y_inspecoes"].to_numpy(dtype=float)

    pack = build_models(random_state=42)
    pack.hgb.fit(X_train, y_train)

    p_val = np.expm1(pack.hgb.predict(X_val))
    p_test = np.expm1(pack.hgb.predict(X_test))
    p_val = np.clip(p_val, 0, None)
    p_test = np.clip(p_test, 0, None)

    return (val, y_val, p_val), (test, y_test, p_test)


def main():
    df = load_hourly_core()
    if df is None or df.empty:
        print("Dataset vazio.")
        return

    df["hora"] = pd.to_datetime(df["hora"], errors="coerce")
    df = df.dropna(subset=["CFGCENTROID", "hora"]).copy()

    # center_id consistente (global)
    df["center_id"] = df["CFGCENTROID"].astype("category").cat.codes.astype(int)

    H = estimate_H_global(df)
    if H <= 0:
        raise SystemExit("Não consegui estimar H (horas operacionais/dia).")

    TEST_H_SHORT = TEST_WEEKS_SHORT * DAYS_PER_WEEK * H
    TEST_H_LONG = TEST_WEEKS_LONG * DAYS_PER_WEEK * H

    # validação: 1 semana (short) e 2 semanas (long) — robusto
    VAL_H_SHORT = TEST_H_SHORT
    VAL_H_LONG = 2 * DAYS_PER_WEEK * H

    centers = sorted(df["CFGCENTROID"].unique())
    print("Nº de centros:", len(centers))
    print("H:", H, "| test_short:", TEST_H_SHORT, "| test_long:", TEST_H_LONG)

    for horizon, test_h, val_h in [
        ("short_1week", TEST_H_SHORT, VAL_H_SHORT),
        ("long_8weeks", TEST_H_LONG, VAL_H_LONG),
    ]:
        min_len = test_h + val_h + MIN_EXTRA_TRAIN_DAYS * H

        # 1) gerar dados de treino do corretor (VAL de todos os centros)
        corr_train_rows = []
        percenter_test_rows = []

        used_centers = 0
        for cid in centers:
            df_c = df[df["CFGCENTROID"] == cid].copy().sort_values("hora")
            if len(df_c) < min_len:
                continue

            sup_c = build_supervised(df_c, target_col="y_inspecoes", time_col="hora")
            out = _train_center_hgb_and_predict(sup_c, test_h=test_h, val_h=val_h)
            if out is None:
                continue

            (val_df, y_val, p_val), (test_df, y_test, p_test) = out
            used_centers += 1

            # features de calendário para o corretor (baseadas em hora)
            val_feat = add_time_features(val_df[["hora"]].copy(), time_col="hora")
            val_feat["CFGCENTROID"] = cid
            val_feat["center_id"] = int(df_c["center_id"].iloc[0])
            val_feat["y_true"] = y_val
            val_feat["pred_center"] = p_val
            corr_train_rows.append(val_feat)

            test_feat = add_time_features(test_df[["hora"]].copy(), time_col="hora")
            test_feat["CFGCENTROID"] = cid
            test_feat["center_id"] = int(df_c["center_id"].iloc[0])
            test_feat["y_true"] = y_test
            test_feat["pred_center"] = p_test
            percenter_test_rows.append(test_feat)

        if used_centers == 0:
            print(f"[{horizon}] Sem centros suficientes.")
            continue

        corr_train = pd.concat(corr_train_rows, ignore_index=True)
        test_all = pd.concat(percenter_test_rows, ignore_index=True)

        # 2) treinar corretor global
        corr_model = fit_global_residual_corrector(corr_train)

        # 3) aplicar ao teste
        test_all["pred_hybrid"] = apply_global_residual_corrector(corr_model, test_all)
        test_all["pred_center_only"] = test_all["pred_center"].to_numpy(dtype=float)

        # 4) métricas por centro + agregadas
        rows = []
        for model_name, pred_col in [("hgb_center", "pred_center_only"), ("hgb_plus_globalcorr", "pred_hybrid")]:
            for cid, g in test_all.groupby("CFGCENTROID"):
                y = g["y_true"].to_numpy(dtype=float)
                p = g[pred_col].to_numpy(dtype=float)
                rows.append(
                    dict(
                        CFGCENTROID=cid,
                        horizon=horizon,
                        model=model_name,
                        mae=mae(y, p),
                        rmse=rmse(y, p),
                        wape=wape(y, p),
                        mape=mape(y, p),
                        mape_pos=mape_pos(y, p),
                        test_sum_y=float(np.sum(y)),
                        test_sum_abs_err=float(np.sum(np.abs(y - p))),
                    )
                )

        res = pd.DataFrame(rows)

        print(f"\n[{horizon}] Centros usados: {used_centers}")
        print("\nBacktest – resumo ponderado e WAPE global real:\n")
        ws = build_weighted_summary(res)
        cols = ["horizon", "model", "mae_mean", "rmse_mean", "wape_mean", "WAPE_global", "mape_mean", "mape_pos_mean", "n_centers"]
        print(ws[cols].round(4).to_string(index=False))

        # Top contribuintes para o erro global (do híbrido)
        sub = res[(res["horizon"] == horizon) & (res["model"] == "hgb_plus_globalcorr")].copy()
        sub["err_contrib"] = sub["test_sum_abs_err"] / (sub["test_sum_abs_err"].sum() + 1e-8)
        top = sub.sort_values("err_contrib", ascending=False).head(10)[
            ["CFGCENTROID", "err_contrib", "wape", "test_sum_y", "test_sum_abs_err"]
        ]
        print("\nTop 10 contribuintes para erro global (híbrido):")
        print(top.round(4).to_string(index=False))

    print("\nDONE.")


if __name__ == "__main__":
    main()