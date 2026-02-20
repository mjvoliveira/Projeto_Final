import numpy as np
import pandas as pd

from src.models_hourly.io import load_hourly_core
from src.models_hourly.models import fit_predict_models, compute_operational_H

DAYS_PER_WEEK = 6
TEST_WEEKS = 8  # foca longo


def sample_param_candidates(rng: np.random.RandomState, n: int = 40) -> list[dict]:
    grid = []
    for lr in [0.01, 0.03, 0.05]:
        for md in [3, 5, 7]:
            for msl in [20, 50, 100]:
                for l2 in [0.0, 0.1, 1.0]:
                    for mi in [600, 900, 1300]:
                        grid.append(
                            dict(
                                loss="absolute_error",
                                learning_rate=lr,
                                max_depth=md,
                                min_samples_leaf=msl,
                                l2_regularization=l2,
                                max_iter=mi,
                            )
                        )
    rng.shuffle(grid)
    return grid[:n]


def eval_params_on_centers(df: pd.DataFrame, centers: list, test_h: int, params: dict):
    sum_y = 0.0
    sum_abs_err = 0.0
    n_ok = 0

    for cid in centers:
        df_c = df[df["CFGCENTROID"] == cid].copy().sort_values("hora")
        res = fit_predict_models(
            df_c,
            test_h=test_h,
            random_state=42,
            hgb_params=params,
        )
        if res is None:
            continue
        if "hgb" not in res["SUMY"]:
            continue

        sum_y += res["SUMY"]["hgb"]
        sum_abs_err += res["SUMABSERR"]["hgb"]
        n_ok += 1

    wape_global = sum_abs_err / (sum_y + 1e-8) if sum_y > 0 else np.nan
    return wape_global, n_ok


if __name__ == "__main__":
    df = load_hourly_core()
    vc = df["CFGCENTROID"].value_counts()
    centers = vc.index[:20].tolist()  # 20 centros com mais dados -> melhor sinal p/ tuning

    # estimar H a partir do primeiro centro "forte"
    df0 = df[df["CFGCENTROID"] == centers[0]].copy().sort_values("hora")
    H = compute_operational_H(df0, time_col="hora")
    test_h = TEST_WEEKS * DAYS_PER_WEEK * H

    rng = np.random.RandomState(123)
    candidates = sample_param_candidates(rng, n=40)

    rows = []
    for i, params in enumerate(candidates):
        wape_g, n_ok = eval_params_on_centers(df, centers, test_h, params)
        rows.append({"candidate": i, "WAPE_global": wape_g, "n_centers": n_ok, "params": params})
        print(f"[{i:02d}] WAPE_global={wape_g:.4f} | n={n_ok} | {params}")

    out = pd.DataFrame(rows).sort_values("WAPE_global")
    print("\nTOP 10 (menor WAPE_global):\n")
    print(out.head(10).to_string(index=False))