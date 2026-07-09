# Correlation Model Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployable Streamlit dashboard that analyzes Baltic P3A_82 against Australia coal shipments, Indonesia coal shipments, and China coal arrivals over the latest 10 years.

**Architecture:** Keep database access, transformations, analytics, and UI in separate modules so tests can run without real credentials. The Streamlit app reads secrets at runtime, queries MySQL through SQLAlchemy, normalizes all series to monthly frequency, and renders Plotly charts plus summary tables.

**Tech Stack:** Python, Streamlit, pandas, numpy, Plotly, SQLAlchemy, PyMySQL, pytest.

---

## File Structure

- `app.py`: Streamlit page layout, sidebar controls, error display, chart/table rendering.
- `requirements.txt`: Runtime and test dependencies for local and Streamlit Cloud.
- `.streamlit/secrets.toml.example`: Safe secrets template for Streamlit Cloud settings.
- `README.md`: Setup, deployment, and Streamlit Cloud secrets instructions.
- `src/__init__.py`: Package marker.
- `src/config.py`: Read and validate Streamlit secrets without exposing passwords.
- `src/data_access.py`: Build SQLAlchemy engines and query AXS/Baltic data.
- `src/transform.py`: Monthly aggregation, joined dataset, indexed values, percentage changes.
- `src/analysis.py`: Pearson/Spearman correlations and lead/lag analysis.
- `src/charts.py`: Plotly trend and heatmap figures.
- `tests/test_transform.py`: Transformation tests using sample data.
- `tests/test_analysis.py`: Correlation and lag-analysis tests using synthetic data.

## Task 1: Project Skeleton And Secrets Contract

**Files:**
- Create: `requirements.txt`
- Create: `.streamlit/secrets.toml.example`
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Test by running: `python -m py_compile src/config.py`

- [ ] **Step 1: Add dependencies**

Create `requirements.txt`:

```txt
streamlit>=1.36
pandas>=2.2
numpy>=1.26
plotly>=5.22
SQLAlchemy>=2.0
PyMySQL>=1.1
pytest>=8.2
```

- [ ] **Step 2: Add Streamlit secrets example**

Create `.streamlit/secrets.toml.example`:

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

- [ ] **Step 3: Add configuration module**

Create `src/config.py` with a `DatabaseSettings` dataclass, `SecretsConfigError`, and `get_database_settings(secrets, section)`.

- [ ] **Step 4: Compile config module**

Run: `python -m py_compile src/config.py`

Expected: no output and exit code 0.

- [ ] **Step 5: Commit skeleton**

Run:

```bash
git add requirements.txt .streamlit/secrets.toml.example README.md src/__init__.py src/config.py
git commit -m "Add Streamlit project skeleton"
```

## Task 2: Transformations With Tests

**Files:**
- Create: `src/transform.py`
- Create: `tests/test_transform.py`
- Test by running: `python -m pytest tests/test_transform.py -q`

- [ ] **Step 1: Write tests**

`tests/test_transform.py` will verify:

- Australia/Indonesia shipments aggregate by `load_start_date`.
- China arrivals aggregate by `discharge_start_date`.
- Monthly Baltic values average by month.
- Joined monthly dataset keeps overlapping months.
- Indexed columns start at 100.
- Month-over-month and year-over-year changes are calculated.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_transform.py -q`

Expected: import failure because `src.transform` does not exist yet.

- [ ] **Step 3: Implement transformation functions**

Create `src/transform.py` with:

- `monthly_volume(df, date_col, value_col, output_col)`
- `monthly_baltic(df, date_col, value_col)`
- `build_monthly_dataset(baltic, australia, indonesia, china)`
- `add_indexed_columns(df, columns)`
- `add_change_columns(df, columns)`

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_transform.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit transformations**

Run:

```bash
git add src/transform.py tests/test_transform.py
git commit -m "Add monthly data transformations"
```

## Task 3: Analysis Model With Tests

**Files:**
- Create: `src/analysis.py`
- Create: `tests/test_analysis.py`
- Test by running: `python -m pytest tests/test_analysis.py -q`

- [ ] **Step 1: Write tests**

`tests/test_analysis.py` will verify:

- Pearson and Spearman correlations are produced for each flow series.
- Insufficient observations return `NaN` correlations and zero observations.
- Lead/lag rows cover every lag from -12 to +12.
- Best lag picks the largest absolute correlation.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_analysis.py -q`

Expected: import failure because `src.analysis` does not exist yet.

- [ ] **Step 3: Implement analysis functions**

Create `src/analysis.py` with:

- `correlation_summary(df, target_col, feature_cols, min_periods=6)`
- `lead_lag_correlations(df, target_col, feature_cols, max_lag=12, min_periods=6)`
- `best_lag_summary(lag_df)`

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_analysis.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit analysis model**

Run:

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "Add correlation analysis model"
```

## Task 4: Database Access Layer

**Files:**
- Create: `src/data_access.py`
- Test by running: `python -m py_compile src/data_access.py`

- [ ] **Step 1: Implement SQLAlchemy engine creation**

Create `create_mysql_engine(settings)` using a URL built from host, database, user, and password.

- [ ] **Step 2: Implement AXS queries**

Create:

- `fetch_australia_shipments(engine, start_date)`
- `fetch_indonesia_shipments(engine, start_date)`
- `fetch_china_arrivals(engine, start_date)`

Each returns only date and volume-relevant columns needed by transformations.

- [ ] **Step 3: Implement Baltic discovery and query**

Create:

- `discover_baltic_p3a82_source(engine)`
- `fetch_baltic_p3a82(engine, start_date)`

The discovery function searches information schema for table/column candidates containing `P3A_82` or `P3A82`, and raises a clear `DataSourceError` if no candidate is found.

- [ ] **Step 4: Compile data access module**

Run: `python -m py_compile src/data_access.py`

Expected: no output and exit code 0.

- [ ] **Step 5: Commit data access**

Run:

```bash
git add src/data_access.py
git commit -m "Add database access layer"
```

## Task 5: Streamlit Dashboard

**Files:**
- Create: `src/charts.py`
- Create: `app.py`
- Test by running: `python -m py_compile app.py src/charts.py`

- [ ] **Step 1: Implement Plotly chart helpers**

Create `src/charts.py` with:

- `trend_figure(df, columns, title)`
- `lag_heatmap(lag_df)`

- [ ] **Step 2: Implement Streamlit app**

Create `app.py` that:

- Reads AXS and Baltic settings from Streamlit secrets.
- Uses a latest-10-years default start date.
- Fetches AXS and Baltic data with cached functions.
- Builds monthly dataset and analysis outputs.
- Renders metric cards, trend figure, lag heatmap, summary table, monthly data table, and CSV download.
- Shows safe error messages for missing secrets, connection errors, empty data, or Baltic discovery failures.

- [ ] **Step 3: Compile UI files**

Run: `python -m py_compile app.py src/charts.py`

Expected: no output and exit code 0.

- [ ] **Step 4: Commit dashboard**

Run:

```bash
git add app.py src/charts.py
git commit -m "Build Streamlit dashboard"
```

## Task 6: Documentation, Verification, Push

**Files:**
- Modify: `README.md`
- Run tests: `python -m pytest -q`
- Push: `git push -u origin main`

- [ ] **Step 1: Complete README**

Document:

- What the model analyzes.
- How to configure Streamlit Cloud secrets.
- How to run locally with optional local secrets.
- Which queries define each series.
- Why monthly aggregation is used.

- [ ] **Step 2: Run full verification**

Run:

```bash
python -m pytest -q
python -m py_compile app.py src/*.py
git status --short
```

Expected:

- Tests pass.
- Compilation passes.
- Only intentional files are modified.

- [ ] **Step 3: Commit documentation**

Run:

```bash
git add README.md docs/superpowers/plans/2026-07-09-correlation-model-dashboard.md
git commit -m "Document correlation model dashboard"
```

- [ ] **Step 4: Push to GitHub**

Run:

```bash
git push -u origin main
```

Expected: branch `main` is pushed to `https://github.com/Ming85671/Correlation-Model.git`.

## Self-Review

- Spec coverage: The plan covers secrets, database access, AXS series definitions, Baltic discovery, monthly transformations, correlation analysis, lead/lag model, Streamlit dashboard, testing, documentation, and GitHub push.
- Placeholder scan: The plan contains no unresolved implementation placeholders.
- Type consistency: Function names introduced in tasks are reused consistently by downstream tasks.
