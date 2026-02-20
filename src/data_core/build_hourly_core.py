from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

# Ajusta aqui para o local real dos teus ficheiros (igual ao mensal)
RAW_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCAO_CSV = RAW_DIR / "Produção.csv"
MARCACOES_CSV = RAW_DIR / "Marcações.csv"

def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def main():
    prod = find_existing([PRODUCAO_CSV, RAW_DIR / "Producao.csv", RAW_DIR / "Producao.csv"])
    marc = find_existing([MARCACOES_CSV, RAW_DIR / "Marcacoes.csv", RAW_DIR / "Marcacões.csv"])

    if prod is None:
        raise SystemExit(f"Não encontrei Produção.csv em {RAW_DIR}. Confirma nomes/paths.")
    if marc is None:
        raise SystemExit(f"Não encontrei Marcações.csv em {RAW_DIR}. Confirma nomes/paths.")

    df_prod = pd.read_csv(prod)
    df_marc = pd.read_csv(marc)

    # --- TODO: AJUSTAR NOMES DE COLUNAS ---
    # Precisamos de: CFGCENTROID, hora (datetime), y_inspecoes
    # Vou fazer um "best effort" com nomes típicos. Se falhar, diz-me as colunas.
    # ------------------------------------

    # 1) tentar localizar coluna de data/hora em produção
    time_candidates = [c for c in df_prod.columns if "data" in c.lower() or "hora" in c.lower() or "timestamp" in c.lower()]
    if not time_candidates:
        raise SystemExit(f"Não encontrei coluna temporal em Produção. Colunas: {list(df_prod.columns)[:50]}")
    time_col = time_candidates[0]

    # 2) centro
    center_candidates = [c for c in df_prod.columns if c.upper() == "CFGCENTROID" or "centro" in c.lower()]
    if not center_candidates:
        raise SystemExit(f"Não encontrei coluna de centro em Produção. Colunas: {list(df_prod.columns)[:50]}")
    center_col = center_candidates[0]

    # 3) target (inspeções) - tenta vários nomes
    y_candidates = [c for c in df_prod.columns if "inspec" in c.lower() or "realiz" in c.lower() or c.lower() in ("y", "y_inspecoes")]
    if not y_candidates:
        raise SystemExit(f"Não encontrei coluna target em Produção. Colunas: {list(df_prod.columns)[:50]}")
    y_col = y_candidates[0]

    df = df_prod[[center_col, time_col, y_col]].copy()
    df.columns = ["CFGCENTROID", "hora", "y_inspecoes"]
    df["hora"] = pd.to_datetime(df["hora"], errors="coerce")
    df = df.dropna(subset=["hora"])

    # Agregar a HORÁRIO (se já estiver horário isto não muda)
    df["hora"] = df["hora"].dt.floor("H")
    df = (
        df.groupby(["CFGCENTROID", "hora"], as_index=False)["y_inspecoes"]
          .sum()
          .sort_values(["CFGCENTROID", "hora"])
    )

    out_path = OUT_DIR / "hourly_all_centers.parquet"
    df.to_parquet(out_path, index=False)
    print("✅ Gravado:", out_path)
    print(df.head())
    print("shape:", df.shape)

if __name__ == "__main__":
    main()