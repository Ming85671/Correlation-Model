# Correlation Model

Streamlit dashboard for comparing the Baltic P3A_82 index with coal shipment and arrival flows from AXS.

The first release analyzes:

- Australia coal shipments
- Indonesia coal shipments
- China coal arrivals excluding China-origin cargo

All series are normalized to monthly frequency over the latest 10 years available.

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

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m pytest -q
```

