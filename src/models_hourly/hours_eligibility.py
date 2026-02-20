import pandas as pd


def compute_hour_eligibility(
    df: pd.DataFrame,
    target_col: str = "y_inspecoes",
    time_col: str = "hora",
    center_col: str = "CFGCENTROID",
    min_positive_rate: float = 0.05,
    min_days: int = 20,
) -> pd.DataFrame:
    """
    Compute hour-of-day eligibility for inspection starts.

    An hour (center, hour_of_day) is eligible if:
      - it appears in at least `min_days`
      - the fraction of days with y > 0 >= min_positive_rate

    Returns a DataFrame with:
      - CFGCENTROID
      - hour_of_day
      - positive_rate
      - n_days
      - hora_elegivel_inicio (bool)
    """

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d["hour_of_day"] = d[time_col].dt.hour
    d["date"] = d[time_col].dt.date

    daily_hour = (
        d.groupby([center_col, "hour_of_day", "date"], as_index=False)[target_col]
         .sum()
    )

    stats = (
        daily_hour
        .assign(has_positive=lambda x: x[target_col] > 0)
        .groupby([center_col, "hour_of_day"], as_index=False)
        .agg(
            n_days=("date", "nunique"),
            n_positive=("has_positive", "sum"),
        )
    )

    stats["positive_rate"] = stats["n_positive"] / stats["n_days"]
    stats["hora_elegivel_inicio"] = (
        (stats["n_days"] >= min_days) &
        (stats["positive_rate"] >= min_positive_rate)
    )

    return stats