# Correlation Model

Streamlit dashboard for comparing the Baltic P3A_82 index with coal shipment and arrival flows from AXS.

The first release analyzes:

- Australia coal shipments
- Indonesia coal shipments
- China coal arrivals excluding China-origin cargo

All series are normalized to monthly frequency over the latest 10 years available.

## Model Scope

The dashboard answers three practical questions:

- How has Baltic P3A_82 moved against Australia coal shipments, Indonesia coal shipments, and China coal arrivals over the last 10 years?
- What is the same-month Pearson and Spearman correlation between Baltic and each coal-flow series?
- Which monthly lead/lag relationship shows the strongest correlation between Baltic and each flow series?

The analysis is descriptive. It is correlation evidence, not a causal model or trading signal.

## Data Definitions

Australia coal shipments:

```sql
SELECT *
FROM axs.axs
WHERE load_country = 'Australia'
  AND voyage_type = 'laden'
  AND COMMODITY LIKE '%COAL%'
ORDER BY load_start_date DESC
```

Indonesia coal shipments:

```sql
SELECT *
FROM axs.axs
WHERE COMMODITY LIKE '%COAL%'
  AND load_country = 'Indonesia'
ORDER BY load_start_date DESC
```

China coal arrivals:

```sql
SELECT *
FROM axs.axs
WHERE COMMODITY LIKE '%COAL%'
  AND discharge_country = 'China'
  AND load_country <> 'China'
ORDER BY discharge_start_date DESC
```

Baltic P3A_82 is discovered from the `market_data` schema by searching for table or column names containing `P3A_82` or `P3A82`.

## Method

- AXS shipment and arrival rows are grouped monthly.
- Baltic P3A_82 is averaged monthly.
- The dashboard uses the overlapping monthly period between all four series.
- Indexed trend charts set each visible series to 100 in the first month.
- Change views show month-over-month and year-over-year percentage changes.
- Lead/lag correlations are calculated from negative to positive monthly lags.

Positive lag means the flow series is shifted forward against Baltic, so the chart compares Baltic with a later flow value. Negative lag means the flow series is shifted backward against Baltic.

## Streamlit Secrets

Do not commit real credentials. In Streamlit Cloud, open the app settings and add:

```toml
[axs]
host = "euwe01prdfrrmsql01.mysql.database.azure.com"
database = "axs"
user = "research_dry"
password = "replace-with-streamlit-cloud-secret"

[baltic]
host = "euwe01prdfrrmsql01.mysql.database.azure.com"
database = "market_data"
user = "marketdatauser"
password = "replace-with-streamlit-cloud-secret"
```

For local development, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the passwords. The local secrets file is ignored by Git.

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. In Streamlit Cloud, create an app from `Ming85671/Correlation-Model`.
3. Set the main file path to `app.py`.
4. Paste the secrets from `.streamlit/secrets.toml.example` into App Settings, replacing only the password values.
5. Deploy the app.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m pytest -q
```

## Project Layout

```text
app.py                  Streamlit dashboard
src/config.py           Streamlit secrets validation
src/data_access.py      MySQL queries and Baltic source discovery
src/transform.py        Monthly aggregation and trend transformations
src/analysis.py         Correlation and lead/lag model
src/charts.py           Plotly chart helpers
tests/                  Unit tests for pure transformation and analysis logic
```
