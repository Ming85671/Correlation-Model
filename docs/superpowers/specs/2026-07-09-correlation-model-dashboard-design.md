# Correlation Model Dashboard Design

## Goal

Build a Streamlit dashboard that compares the 10-year movement of the Baltic P3A_82 index with coal trade volume changes from AXS:

- Australia coal shipments
- Indonesia coal shipments
- China coal arrivals, excluding China-origin cargo

The first release should answer whether these trade-flow series move with, lead, or lag the Baltic P3A_82 index.

## Data Sources

The app will read from two MySQL databases through Streamlit Secrets.

AXS database:

- Host: `euwe01prdfrrmsql01.mysql.database.azure.com`
- Database: `axs`
- Table: `axs.axs`
- User: `research_dry`

Baltic database:

- Host: `euwe01prdfrrmsql01.mysql.database.azure.com`
- Database: `market_data`
- User: `marketdatauser`

Secrets will not be committed to Git. Local development may use `.streamlit/secrets.toml`; Streamlit Cloud will use its App Settings secrets.

## AXS Series Definitions

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

India coal arrivals are out of scope for the first dashboard, but the query pattern should be easy to add later.

## Baltic Series Definition

The app will inspect the `market_data` schema to locate the P3A_82 date and value fields, then query the 10-year Baltic P3A_82 index history.

Because the exact Baltic table name and column names have not been confirmed yet, the implementation will isolate this behind a data-access function. If the table cannot be detected automatically, the app will show a clear configuration error instead of failing silently.

## Transformations

The app will normalize all series to monthly frequency:

- Shipment and arrival volumes aggregate by month.
- Baltic P3A_82 aggregates by monthly average.
- The default date range is the latest 10 years available after joining the series.
- Trend charts use indexed values with the first visible month set to 100.
- Change charts use month-over-month and year-over-year percentage changes.

The first release will use monthly aggregation to reduce daily noise and avoid misleading correlations from mismatched date granularity.

## Analysis

The model will calculate:

- Pearson correlation between Baltic P3A_82 and each coal-flow series.
- Spearman correlation as a rank-based robustness check.
- Lead/lag correlation from -12 to +12 months for each coal-flow series.
- Best lag per series, defined as the lag with the largest absolute Pearson correlation.

Lag interpretation will be explicit:

- Positive lag means the coal-flow series is shifted forward against Baltic, implying Baltic may lead that flow series.
- Negative lag means the coal-flow series is shifted backward against Baltic, implying the flow series may lead Baltic.

The dashboard will label this as correlation evidence, not causation.

## Dashboard Layout

Use the Analysis Cockpit layout.

Top section:

- Latest Baltic P3A_82 value and recent change.
- Correlation card for Australia coal shipments.
- Correlation card for Indonesia coal shipments.
- Correlation card for China coal arrivals.

Main section:

- 10-year indexed trend chart showing Baltic P3A_82, Australia shipments, Indonesia shipments, and China arrivals.
- Optional selector for absolute volume, indexed level, month-over-month change, and year-over-year change.

Lead/lag section:

- Heatmap of lag correlations from -12 to +12 months.
- Summary table with Pearson, Spearman, best lag, best-lag correlation, and observation count.

Data section:

- Monthly joined dataset preview.
- CSV download for the transformed analysis table.
- Refresh timestamp and data source status.

## Error Handling

The app should handle common deployment failures clearly:

- Missing Streamlit secrets.
- MySQL connection failure.
- Baltic P3A_82 table or fields not found.
- Missing required AXS columns.
- Empty result sets after filtering.
- Too few overlapping monthly observations for correlation.

The app should avoid exposing passwords or raw connection strings in error messages.

## Testing

Automated tests will focus on pure transformation and analysis logic:

- Monthly aggregation from sample AXS rows.
- Indexed trend calculation.
- Month-over-month and year-over-year changes.
- Pearson and Spearman correlation output.
- Lead/lag correlation direction and best-lag selection.
- Defensive behavior for empty or insufficient datasets.

Database integration will be structured so tests can run without real credentials.

## Deployment

The repository will be pushed to:

`https://github.com/Ming85671/Correlation-Model.git`

Streamlit Cloud deployment will use:

- GitHub repository: `Ming85671/Correlation-Model`
- Main app file: `app.py`
- Secrets configured in Streamlit Cloud App Settings

The committed repository will include a `.streamlit/secrets.toml.example` file, but not real credentials.

## Out Of Scope For First Release

- India coal arrivals.
- Vessel-class segmentation.
- Route-level analysis.
- Forecasting or predictive trading signals.
- Automated daily report generation.
- Persisting query results outside Streamlit cache.
