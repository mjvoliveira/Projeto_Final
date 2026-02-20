# 📊 Projeto de Previsão de Procura – Controlauto

## 📌 Visão Geral

Este projeto implementa um sistema de previsão de procura para centros de inspeção automóvel, com modelação em três níveis de granularidade:

- 📅 **Mensal** – Planeamento estratégico (até 12 meses)
- 📆 **Semanal** – Planeamento tático (até 2 meses)
- 📍 **Diário** – Planeamento operacional (até 2 semanas)

Cada centro é modelado individualmente, captando padrões específicos de sazonalidade, tendência e comportamento operacional.

---

## 📁 Estrutura do Projeto

```
data/
 ├── raw/              # Dados brutos (não versionados)
 └── processed/        # Dados agregados (Parquet)

src/
 ├── models_monthly/
 ├── models_weekly/
 └── models_daily/
```

---

## 📂 Dados

Os ficheiros CSV em `data/raw/` **não estão versionados** devido à sua dimensão.

Para executar o projeto, devem ser colocados manualmente na pasta:

```
data/raw/
```

Ficheiros necessários:

- `Producao.csv`
- `Marcacoes.csv`
- `Slotsdisponiveis.csv`

---

# 🔵 1. Previsões Mensais

## Objetivo

Prever o número de inspeções por centro com horizonte até **12 meses**, suportando planeamento estratégico e alocação de recursos.

## Dados

Agregação mensal dos registos de produção:

| Coluna        | Descrição |
|--------------|-----------|
| CFGCENTROID  | Identificador do centro |
| MONTH        | Mês de referência |
| y            | Nº de inspeções no mês |

Os dados processados são armazenados em formato **Parquet** para maior eficiência.

## Metodologia

- Modelo: **Holt-Winters aditivo**
- Sazonalidade: 12 meses
- Modelação individual por centro
- Comparação com baseline e SARIMA

## Validação

Backtesting one-step-ahead:

- Treino até penúltimo mês
- Previsão do último mês observado

### Métricas

- MAE  
- RMSE  
- WAPE  

Erro médio observado: **~2–3% do volume mensal por centro**

## Outputs

- `forecast_monthly_all_centers.parquet`
- Gráficos histórico vs previsão
- Métricas agregadas por centro

## Execução

```bash
python src/models_monthly/backtest_all_centers.py
```

---

# 🟠 2. Previsões Semanais

## Objetivo

Prever o número de inspeções por centro com horizonte até **2 meses**, suportando planeamento tático e ajuste de capacidade.

## Dados

Agregação semanal da produção histórica.

| Coluna        | Descrição |
|--------------|-----------|
| CFGCENTROID  | Centro |
| WEEK         | Semana de referência |
| y            | Nº de inspeções na semana |

## Metodologia

- Modelo: **HistGradientBoostingRegressor**
- Modelação individual por centro
- Features temporais:
  - Semana do ano
  - Mês
  - Ano

## Validação

Backtesting com janela temporal deslizante (rolling window).

### Métricas

- MAE  
- RMSE  
- WAPE  

Erro médio semanal observado: **~6%**

## Outputs

- `metrics_semanal.parquet`
- `dados_semanal.parquet`
- Gráficos de rolling backtest

## Execução

```bash
python src/models_weekly/backtest_weekly_all_centers.py
```

---

# 🔴 3. Previsões Diárias

## Objetivo

Prever o número de inspeções por centro com horizonte até **2 semanas**, suportando planeamento operacional diário.

## Dados

Agregação diária da produção:

| Coluna        | Descrição |
|--------------|-----------|
| CFGCENTROID  | Centro |
| data         | Data |
| y_inspecoes  | Nº inspeções |

## Metodologia

Modelo de Machine Learning por centro:

- **HistGradientBoostingRegressor**
- Modelação individual por centro
- Features temporais:
  - Dia da semana (`dow`)
  - Semana do ano
  - Mês
  - Ano
  - Indicador de sábado
- Variáveis operacionais:
  - `is_holiday`
  - `is_pre_holiday` (véspera de feriado)

## Validação

Backtesting temporal (últimos 3 meses):

- Treino histórico
- Avaliação diária por centro

### Métricas

- MAE  
- RMSE  
- WAPE  

Erro mediano diário observado: **~12–13%**

## Outputs

- `metrics_diario.parquet`
- `dados_diario.parquet`
- Gráficos de rolling backtest (14 dias)
- Forecast de 2 semanas futuras

## Execução

```bash
python src/models_daily/backtest_daily_all_centers.py
```

---

# 📈 Comparação de Performance

| Granularidade | Horizonte | WAPE Médio | Utilização |
|--------------|------------|------------|------------|
| Mensal       | 12 meses   | ~2–3%      | Estratégico |
| Semanal      | 2 meses    | ~6%        | Tático |
| Diário       | 2 semanas  | ~12–13%    | Operacional |

A granularidade diária apresenta maior volatilidade, mas mantém erro controlado para suporte a decisões de curto prazo.

---

# ⚙️ Dependências

- Python 3.10+
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- pyarrow (para Parquet)

Instalação recomendada:

```bash
pip install -r requirements.txt
```

---

# 🚧 Limitações

- Modelo univariado no nível mensal
- Eventos operacionais extraordinários não estruturados não são diretamente previsíveis
- Dependência da qualidade e consistência dos registos históricos
- Sensibilidade do modelo diário a dias atípicos (ex.: encerramentos excecionais)

---

# 🔮 Evolução Futura

- Integração de variáveis exógenas adicionais (campanhas, capacidade, encerramentos programados)
- Monitorização automática de drift
- API para consulta interativa por centro
- Automação de pipelines de treino e forecast

---

# 📌 Conclusão

O projeto demonstra robustez na previsão multi-nível da procura, permitindo transformar dados históricos em suporte estruturado à decisão estratégica, tática e operacional.
