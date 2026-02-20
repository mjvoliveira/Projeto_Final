import pandas as pd
from pathlib import Path

DATA_PATH = Path('/Users/catarinatorres/Documents/ControlAuto/Data/Raw/controlauto_outputs/data_core_hourly_v1.csv')


def load_hourly_core(
    filter_closed_hours: bool = False,  # mantido para compatibilidade
) -> pd.DataFrame:
    """
    Carrega o core horário e normaliza timestamps para a hora cheia.

    - lê CSV
    - converte "hora" para datetime
    - floor para hora cheia (ex.: 15:58:42 -> 15:00:00)
    - agrega duplicados por (CFGCENTROID, hora)
    - ordena

    Nota: filter_closed_hours fica aqui só por compatibilidade; não é aplicado neste loader.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Não encontrei o ficheiro: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"CFGCENTROID", "hora", "y_inspecoes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no dataset: {sorted(missing)}")

    df["hora"] = pd.to_datetime(df["hora"], errors="coerce")
    df = df.dropna(subset=["hora"])

    # Normalização horária (crítico para lags por timestamp)
    df["hora"] = df["hora"].dt.floor("H")

    # Agregar duplicados por centro/hora
    df = (
        df.groupby(["CFGCENTROID", "hora"], as_index=False)
        .agg(y_inspecoes=("y_inspecoes", "sum"))
    )

    df = df.sort_values(["CFGCENTROID", "hora"]).reset_index(drop=True)

    _ = filter_closed_hours
    return df