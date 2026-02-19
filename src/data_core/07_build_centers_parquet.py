from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[2]  # raiz do projeto
RAW_PATH = BASE_PATH / "data" / "raw"
PROCESSED_PATH = BASE_PATH / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

IN_XLSX = RAW_PATH / "Coordenadas centros.xlsx"
OUT_PARQUET = PROCESSED_PATH / "centers.parquet"

df = pd.read_excel(IN_XLSX)

df = df.rename(columns={
    "# Centro": "center_id",
    "Designação Centro (Base IMT, 2013)": "center_name",
})

df["center_id"] = df["center_id"].astype(str).str.strip()
df["center_name"] = df["center_name"].astype(str).str.strip()

df = df.dropna(subset=["center_id"])
df = df.drop_duplicates(subset=["center_id"], keep="first")

df["label"] = df["center_id"] + " - " + df["center_name"]

df[["center_id", "center_name", "label"]].to_parquet(OUT_PARQUET, index=False)
print(f"OK -> {OUT_PARQUET} | rows={len(df)}")
