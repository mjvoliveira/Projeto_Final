import argparse
import numpy as np
import pandas as pd

from src.models_hourly.io import load_hourly_core
from src.models_hourly.models import (
    fit_predict_models,
    compute_operational_H,
    predict_test_window,
)

DAYS_PER_WEEK = 6
TEST_WEEKS_SHORT = 1
TEST_WEEKS_LONG = 8
MIN_EXTRA_TRAIN_DAYS = 30


def estimate_H_global(df: pd.DataFrame) -> int:
    # usa o centro com mais registos só para estimar o nº de horas operacionais por dia (H)
    top_center = df["CFGCENTROID"].value_counts().index[0]
    g = df[df["CFGCENTROID"] == top_center].copy()
    return compute_operational_H(g, time_col="hora")


def build_weighted_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumo por (horizon, model) com:
      - médias simples
      - MAE ponderado por volume (sum_y)
      - WAPE global real = sum(abs_err)/sum(y)
      - MAPE médio (cuidado com zeros)
      - MAPE_pos médio (ignora y=0) -> normalmente o que queres interpretar
    """
    if results_df.empty:
        return pd.DataFrame()

    g = results_df.groupby(["horizon", "model"], as_index=False)

    out = g.agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        wape_mean=("wape", "mean"),
        mape_mean=("mape", "mean"),
        mape_pos_mean=("mape_pos", "mean"),
        MAE_weighted_sumy=("mae_weighted_sumy", "sum"),
        sum_y=("test_sum_y", "sum"),
        sum_abs_err=("test_sum_abs_err", "sum"),
        n_centers=("CFGCENTROID", "nunique"),
    )

    out["WAPE_global"] = out["sum_abs_err"] / (out["sum_y"] + 1e-8)

    # limpeza
    out = out.drop(columns=["sum_y", "sum_abs_err"])
    out = out.sort_values(["horizon", "WAPE_global", "mae_mean"]).reset_index(drop=True)
    return out


def plot_test_vs_pred(df_c: pd.DataFrame, test_h: int, model: str = "hgb", title_extra: str = ""):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib não disponível para plot.")
        return

    out = predict_test_window(df_c, test_h=test_h, model_name=model)
    if out is None or out.empty:
        print("Sem previsão para plot (série curta / desalinhada).")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(out["hora"], out["y_true"], label="y_true")
    plt.plot(out["hora"], out["y_pred"], label=f"y_pred ({model})")
    plt.title(f"Centro {df_c['CFGCENTROID'].iloc[0]} | {title_extra}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_center", type=str, default=None, help="CFGCENTROID para plot")
    parser.add_argument("--plot_model", type=str, default="hgb", help="modelo p/ plot (hgb/ridge/mean_by_hour_dow/ens_hgb_mean)")
    parser.add_argument("--plot_horizon", type=str, default="long_8weeks", choices=["short_1week", "long_8weeks"])
    args = parser.parse_args()

    df = load_hourly_core()
    if df.empty:
        print("Dataset vazio.")
        return

    H_global = estimate_H_global(df)
    if H_global <= 0:
        print("Não consegui estimar H global.")
        return

    TEST_HOURS_SHORT = TEST_WEEKS_SHORT * DAYS_PER_WEEK * H_global
    TEST_HOURS_LONG = TEST_WEEKS_LONG * DAYS_PER_WEEK * H_global

    centers = df["CFGCENTROID"].dropna().unique().tolist()
    centers = sorted(centers)

    rows = []

    for cid in centers:
        df_c = df[df["CFGCENTROID"] == cid].copy().sort_values("hora")
        if df_c.empty:
            continue

        # só corre se houver margem de treino
        if len(df_c) < TEST_HOURS_LONG + MIN_EXTRA_TRAIN_DAYS * max(H_global, 1):
            # deixa passar centros curtos
            pass

        for horizon, test_h in [("short_1week", TEST_HOURS_SHORT), ("long_8weeks", TEST_HOURS_LONG)]:
            res = fit_predict_models(df_c, test_h=test_h, random_state=42)
            if res is None:
                continue

            # por cada modelo devolvido
            for model_name in res["MAE"].keys():
                y_sum = res["SUMY"][model_name]
                mae_val = res["MAE"][model_name]
                rmse_val = res["RMSE"][model_name]
                wape_val = res["WAPE"][model_name]
                mape_val = res["MAPE"][model_name]
                mape_pos_val = res["MAPE_POS"][model_name]

                rows.append(
                    dict(
                        CFGCENTROID=cid,
                        horizon=horizon,
                        model=model_name,
                        mae=mae_val,
                        rmse=rmse_val,
                        wape=wape_val,
                        mape=mape_val,
                        mape_pos=mape_pos_val,
                        test_sum_y=float(res["SUMY"][model_name]),
                        test_sum_abs_err=float(res["SUMABSERR"][model_name]),
                        mae_weighted_sumy=float(mae_val * y_sum),
                    )
                )

    results_df = pd.DataFrame(rows)
    if not len(results_df):
        print("Sem resultados (séries curtas ou algo não alinhado).")
        return

    # ---- tabelas
    print("\nBacktest – resumo simples (média por centro):\n")
    summary = (
        results_df.groupby(["horizon", "model"], as_index=False)[["mae", "rmse", "wape", "mape", "mape_pos"]]
        .mean()
        .sort_values(["horizon", "mae"])
    )
    print(summary.round(4).to_string(index=False))

    print("\nBacktest – resumo ponderado e WAPE global real:\n")
    ws = build_weighted_summary(results_df)
    if len(ws):
        cols = [
            "horizon",
            "model",
            "mae_mean",
            "rmse_mean",
            "wape_mean",
            "WAPE_global",
            "mape_mean",
            "mape_pos_mean",
            "MAE_weighted_sumy",
            "n_centers",
        ]
        print(ws[cols].round(4).to_string(index=False))
    else:
        print("Sem resumo ponderado.")

    # ---- concentração de erro: top centros (pior WAPE e maior contribuição para erro global)
    if len(results_df) and len(ws):
        for hz in ["short_1week", "long_8weeks"]:
            sub_ws = ws[ws["horizon"] == hz].sort_values("WAPE_global")
            if sub_ws.empty:
                continue
            best_model = sub_ws.iloc[0]["model"]
            print(f"\n[Concentração de erro] horizonte={hz} | melhor_modelo={best_model}")

            sub = results_df[(results_df["horizon"] == hz) & (results_df["model"] == best_model)].copy()
            if sub.empty:
                continue

            sub["center_WAPE"] = sub["test_sum_abs_err"] / (sub["test_sum_y"] + 1e-8)

            total_abs_err = sub["test_sum_abs_err"].sum()
            sub["err_contrib"] = sub["test_sum_abs_err"] / (total_abs_err + 1e-8)

            top_wape = sub.sort_values("center_WAPE", ascending=False).head(10)[
                ["CFGCENTROID", "center_WAPE", "test_sum_y", "test_sum_abs_err", "err_contrib"]
            ]
            print("\nTop 10 piores centros por WAPE:")
            print(top_wape.round(4).to_string(index=False))

            top_contrib = sub.sort_values("err_contrib", ascending=False).head(10)[
                ["CFGCENTROID", "err_contrib", "center_WAPE", "test_sum_y", "test_sum_abs_err"]
            ]
            print("\nTop 10 centros que mais contribuem para o erro global:")
            print(top_contrib.round(4).to_string(index=False))

    # ---- optional plot
    if args.plot_center is not None:
        cid = args.plot_center
        if cid.isdigit():
            cid = int(cid)

        df_c = df[df["CFGCENTROID"] == cid].copy().sort_values("hora")
        if df_c.empty:
            print(f"Centro {cid} não encontrado.")
        else:
            test_h = TEST_HOURS_LONG if args.plot_horizon == "long_8weeks" else TEST_HOURS_SHORT
            plot_test_vs_pred(df_c, test_h=test_h, model=args.plot_model, title_extra=args.plot_horizon)


if __name__ == "__main__":
    main()