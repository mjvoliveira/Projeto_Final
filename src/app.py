# src/app.py
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PROCESSED_PATH = BASE_PATH / "data" / "processed"

CENTERS_PARQUET = DATA_PROCESSED_PATH / "centers.parquet"
LOGO_PATH = BASE_PATH / "data" / "features" / "controlauto_logo.svg"

# Mensal
METRICS_MONTHLY_PATH = DATA_PROCESSED_PATH / "monthly_metrics_last12_all_centers.parquet"
DATA_MONTHLY_PATH = DATA_PROCESSED_PATH / "monthly_forecast_12m_all_centers.parquet"

# Semanal
METRICS_WEEKLY_PATH = DATA_PROCESSED_PATH / "metrics_semanal.parquet"
DATA_WEEKLY_PATH = DATA_PROCESSED_PATH / "dados_semanal.parquet"

# Diário
METRICS_DAILY_PATH = DATA_PROCESSED_PATH / "metrics_diario.parquet"
DATA_DAILY_PATH = DATA_PROCESSED_PATH / "dados_diario.parquet"

PERIOD_MAP = {
    "Mensal": {"metrics": METRICS_MONTHLY_PATH, "data": DATA_MONTHLY_PATH},
    "Semanal": {"metrics": METRICS_WEEKLY_PATH, "data": DATA_WEEKLY_PATH},
    "Diário": {"metrics": METRICS_DAILY_PATH, "data": DATA_DAILY_PATH},
}


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    return pf.read().to_pandas()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(page_title="Forecast de Inspeções", layout="wide")

# -----------------------------
# Header com logo
# -----------------------------
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=250)

with col_title:
    st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        text-transform: uppercase;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        margin-bottom: 5px;
    }
    .custom-divider {
        height: 4px;
        background-color: #F37021;
        border-radius: 1px;
        margin-top: 1px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 28px;
        font-weight: 600;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">Forecast de Inspeções</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# 1) Carregar centers.parquet (só isto antes das dropdowns)
# -----------------------------
if not CENTERS_PARQUET.exists():
    st.error(f"Ficheiro em falta: {CENTERS_PARQUET}")
    st.stop()

centros_df = normalize_cols(load_parquet(str(CENTERS_PARQUET)))

# Validar colunas mínimas do centers.parquet
if "center_id" not in centros_df.columns or "label" not in centros_df.columns:
    st.error(f"centers.parquet precisa de colunas ['center_id','label']. Tenho: {list(centros_df.columns)}")
    st.stop()


# -----------------------------
# 2) Dropdowns lado a lado (como 1ª versão)
#    - Centro à esquerda
#    - Periodicidade à direita
#    - Mensal pré-selecionado
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-title">Seleciona um Centro</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">Escolhe a Periodicidade</div>', unsafe_allow_html=True)

# Periodicidade primeiro (para sabermos que ficheiros ler e filtrarmos centros com dados)
with col2:
    periodicidade = st.selectbox("", ["Mensal", "Semanal", "Diário"], index=0)

paths = PERIOD_MAP[periodicidade]


# -----------------------------
# 3) Validação + carregar metrics/data + normalização de colunas
# -----------------------------
required_files = [paths["metrics"], paths["data"]]
missing = [p for p in required_files if not p.exists()]
if missing:
    st.error("Ficheiros em falta:\n\n" + "\n".join([str(p) for p in missing]))
    st.stop()

metrics_df = normalize_cols(load_parquet(str(paths["metrics"])))
data_df = normalize_cols(load_parquet(str(paths["data"])))

# Normalizar coluna do centro (caso venha como center_id)
if "cfgcentroid" not in metrics_df.columns and "center_id" in metrics_df.columns:
    metrics_df = metrics_df.rename(columns={"center_id": "cfgcentroid"})
if "cfgcentroid" not in data_df.columns and "center_id" in data_df.columns:
    data_df = data_df.rename(columns={"center_id": "cfgcentroid"})

# Validar que cfgcentroid existe
if "cfgcentroid" not in metrics_df.columns:
    st.error(f"metrics_df sem coluna 'cfgcentroid'. Colunas: {list(metrics_df.columns)}")
    st.stop()
if "cfgcentroid" not in data_df.columns:
    st.error(f"data_df sem coluna 'cfgcentroid'. Colunas: {list(data_df.columns)}")
    st.stop()

# Filtrar centros para mostrar só os que têm dados nesta periodicidade
centros_com_dados = set(data_df["cfgcentroid"].dropna().astype(int).unique())

centros_df["center_id"] = centros_df["center_id"].astype(str)
centros_df["center_id_num"] = pd.to_numeric(centros_df["center_id"], errors="coerce")

centros_df = centros_df[centros_df["center_id_num"].isin(centros_com_dados)].copy()
centros_df = centros_df.sort_values(["center_id_num", "center_id"], na_position="last")

# Agora sim: dropdown do centro (à esquerda), já filtrado pelos centros com dados
with col1:
    centro_label = st.selectbox("", centros_df["label"].tolist(), index=0)

center_id = centro_label.split(" - ")[0].strip()

# -----------------------------
# Views (filtrar dados do centro selecionado)
# -----------------------------
metrics_view = metrics_df[metrics_df["cfgcentroid"].astype(str) == str(center_id)].copy()
data_view = data_df[data_df["cfgcentroid"].astype(str) == str(center_id)].copy()

st.divider()

# -----------------------------
# Título Centro
# -----------------------------
center_name_only = centro_label.split(" - ", 1)[1]

st.markdown(
    f"""
    <div style="
        font-size:36px;
        font-weight:700;
        color:#9DB3BF;
        text-align:center;
        margin-bottom:4px;
    ">
        {center_name_only}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="
        font-size:18px;
        font-weight:500;
        color:#9DB3BF;
        text-align:center;
        margin-bottom:28px;
    ">
        Forecast {periodicidade}
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# KPI contexto (período + modelo)
# -----------------------------
if periodicidade == "Mensal":
    periodo_txt = "Período avaliado: últimos 12 meses"
    modelo_txt = "Modelo: Holt-Winters (pré-selecionado)"  # ajusta para o vosso modelo real
elif periodicidade == "Semanal":
    periodo_txt = "Período avaliado: últimas 8 semanas (~2 meses)"
    modelo_txt = "Modelo: HistGradientBoostingRegressor (pré-selecionado)"  # ajusta para o vosso modelo real
else:  # Diário
    periodo_txt = "Período avaliado: últimos 14 dias (~2 semanas)"
    modelo_txt = "Modelo: HistGradientBoostingRegressor (pré-selecionado)"  # ajusta para o vosso modelo real



# -----------------------------
# Métricas
# -----------------------------
st.markdown(f"### Métricas")

with st.container(border=True):
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if metrics_view.empty:
        st.warning("Sem métricas para este centro.")
    else:
        row = metrics_view.iloc[0]

        mae = float(row["mae"]) if "mae" in metrics_view.columns else None
        rmse = float(row["rmse"]) if "rmse" in metrics_view.columns else None
        wape = float(row["wape"]) if "wape" in metrics_view.columns else None

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE - Erro médio absoluto", f"{mae:.2f}" if mae else "—")
        c2.metric("RMSE - Raiz erro quadratico médio", f"{rmse:.2f}" if rmse else "—")
        c3.metric("WAPE - Percentagem erro absoluto", f"{wape:.2%}" if wape else "—")
    
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="
        font-size:14px;
        color:#6B7280;
        margin-top:-6px;
        margin-bottom:12px;
    ">
        {periodo_txt} &nbsp;&nbsp;|&nbsp;&nbsp; {modelo_txt}
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Forecast (Histórico + Projeção)
# -----------------------------
st.markdown(f"### Forecast")



if data_view.empty:
    st.warning("Sem dados para este centro.")
else:
    date_col = None
    for c in ["month", "date", "day", "week", "timestamp"]:
        if c in data_view.columns:
            date_col = c
            break

    if date_col is None:
        st.stop()

    data_view[date_col] = pd.to_datetime(data_view[date_col], errors="coerce")
    data_view["value"] = pd.to_numeric(data_view["value"], errors="coerce").round(0)
    data_view["kind"] = data_view["kind"].astype(str).str.lower()

    data_view = data_view.dropna(subset=[date_col]).sort_values(date_col)

    data_view["tipo"] = data_view["kind"].replace({
        "history": "Histórico",
        "history": "Histórico",
        "forecast": "Forecast",
        "projecao": "Forecast",
        "projeção": "Forecast"
    })

    fc_plot = data_view[[date_col, "value", "tipo"]].rename(
        columns={date_col: "data", "value": "valor"}
    )

    color_scale = alt.Scale(
        domain=["Histórico", "Forecast"],
        range=["#1f77b4", "#F37021"]
    )


# -----------------------------
# Preparação do eixo X (robusta e igual à tabela)
# -----------------------------
fc_plot = fc_plot.copy()
fc_plot["data_dt"] = pd.to_datetime(fc_plot["data"])

if periodicidade == "Mensal":
    # parecido com tabela: "Feb 2026" (ajusta se quiseres PT)
    fc_plot["data_label"] = fc_plot["data_dt"].dt.strftime("%b %Y")
    tick_count = 6

elif periodicidade == "Semanal":
    # parecido com tabela: "Sem 07 - 2026"
    fc_plot["data_label"] = fc_plot["data_dt"].dt.strftime("Sem %W - %Y")
    tick_count = 8

else:  # Diário
    # parecido com tabela: "18/02/2026"
    fc_plot["data_label"] = fc_plot["data_dt"].dt.strftime("%d/%m/%Y")
    tick_count = 10

x_axis = alt.Axis(labelAngle=0)

# -----------------------------
# Gráfico
# -----------------------------
chart = (
    alt.Chart(fc_plot)
    .mark_line(point=True)
    .encode(
        # eixo X como ordinal (label já formatada), mas ordenado por datetime real
        x=alt.X(
            "data_label:O",
            title="Data",
            sort=alt.SortField(field="data_dt", order="ascending"),
            axis=x_axis,
        ),
        y=alt.Y("valor:Q", title="Inspeções"),
        color=alt.Color("tipo:N", scale=color_scale, legend=alt.Legend(title="Série")),
        tooltip=[
            alt.Tooltip("data_dt:T", title="Data"),
            alt.Tooltip("tipo:N", title="Tipo"),
            alt.Tooltip("valor:Q", title="Valor"),
        ],
    )
    .properties(height=340)
)

st.altair_chart(chart, use_container_width=True)


# -----------------------------
# Legenda dinâmico Forecast
# -----------------------------
if periodicidade == "Mensal":
    hist_txt = "Período de histórico: últimos 12 meses"
    fc_txt   = "Período de forecast: 12 meses"

elif periodicidade == "Semanal":
    hist_txt = "Período de histórico: últimos 2 meses"
    fc_txt   = "Período de forecast: 2 meses"

else:  # Diário
    hist_txt = "Período de histórico: últimos 2 semanas"
    fc_txt   = "Período de forecast: 2 semanas"

st.markdown(
    f"""
    <div style="
        font-size:14px;
        color:#6B7280;
        margin-top:-6px;
        margin-bottom:18px;
    ">
        {hist_txt} &nbsp;&nbsp;|&nbsp;&nbsp; {fc_txt}
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Tabela (escondida)
# -----------------------------
with st.expander("Ver tabela detalhada"):
    
    tabela = fc_plot.copy()

    # Data curta dependendo da periodicidade
    if periodicidade == "Mensal":
        tabela["data"] = pd.to_datetime(tabela["data"]).dt.strftime("%b %Y")
    elif periodicidade == "Semanal":
        tabela["data"] = pd.to_datetime(tabela["data"]).dt.strftime("Sem %W - %Y")
    elif periodicidade == "Diário":
        tabela["data"] = pd.to_datetime(tabela["data"]).dt.strftime("%d/%m/%Y")

    # Garantir inteiro limpo
    tabela["valor"] = tabela["valor"].astype("Int64")

    st.dataframe(
        tabela.rename(columns={
            "data": "Data",
            "valor": "Inspeções",
            "tipo": "Tipo"
        }).reset_index(drop=True),
        use_container_width=True
    )

