# Correlation Model

Streamlit dashboard for comparing the Baltic P3A_82 index with coal shipment and arrival flows from AXS.

The first release analyzes:

- Australia coal shipments
- Indonesia coal shipments
- China coal arrivals excluding China-origin cargo

The dashboard supports monthly and daily correlation views over the selected history. For every
origin/destination flow, choose either shipment count or cargo volume before reading a correlation.

## Model Scope

The dashboard answers three practical questions:

- How has Baltic P3A_82 moved against Australia coal shipments, Indonesia coal shipments, and China coal arrivals over the last 10 years?
- What is the same-month Pearson and Spearman correlation between Baltic and each coal-flow series?
- Which of the automatically ranked top 10 lead/lag relationships is strongest across Baltic and the flow series?
- Does P3A move in the same or opposite direction as daily or monthly cargo levels?

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

- Shipment count: count of qualifying AXS shipment records (one voyage/ship record in the source).
- Cargo volume: sum of the available AXS volume field for the same records.

Indonesia coal shipments:

```sql
SELECT *
FROM axs.axs
WHERE COMMODITY LIKE '%COAL%'
  AND load_country = 'Indonesia'
ORDER BY load_start_date DESC
```

- Shipment count: count of qualifying AXS shipment records.
- Cargo volume: sum of the available AXS volume field.

China coal arrivals:

```sql
SELECT *
FROM axs.axs
WHERE COMMODITY LIKE '%COAL%'
  AND discharge_country = 'China'
  AND load_country <> 'China'
ORDER BY discharge_start_date DESC
```

- Arrival count: count of qualifying AXS arrival records.
- Arrival volume: sum of the available AXS volume field.

Baltic P3A_82 is discovered from the `market_data` schema by searching for table or column names containing `P3A_82` or `P3A82`.

## Method

- Monthly mode groups AXS shipment and arrival rows by month and averages Baltic P3A_82 by month.
- Daily mode retains calendar-day cargo totals, including zero-flow days. Baltic stays missing on
  weekends and market holidays rather than being forward-filled.
- Daily correlation compares P3A levels with daily cargo levels. Daily trend charts standardize
  each series so their movements can be compared on one scale.
- Shipment count and cargo volume are independent measures: a missing volume field remains missing
  and is never substituted with the shipment count.
- The dashboard uses the overlapping monthly period between all four series.
- Indexed trend charts set each visible series to 100 in the first month.
- Change views show month-over-month and year-over-year percentage changes.
- Lead/lag correlations are calculated over the full automatic range of plus/minus 24 months in
  monthly mode and plus/minus 60 calendar days in daily mode, then ranked by absolute Pearson
  correlation across every flow and lag.
- The minimum reliability threshold is automatic: it uses half of the smallest usable paired
  sample, with a 60-day or 12-month floor whenever the available data can support that floor. It
  is never chosen to maximize a correlation coefficient.

Positive lag means cargo occurs first and P3A is compared with a later value. Negative lag means
P3A occurs first and is compared with a later cargo value.

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
