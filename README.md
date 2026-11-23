# 🌊 TDA River Analysis

Streamlit app for Topological Data Analysis of river water-level data:
time-delay embedding, persistent homology, ToMATo clustering, state
identification, transition networks and sliding-window L2 norm.

## Input data
Upload a CSV via the sidebar. Expected columns (in order):
`Date`, `Raw`, `Level` (dates parsed day-first). With only 2 columns,
the first is used as Date and the last as Level.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **Create app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings**, choose Python 3.11 or 3.12, then **Deploy**.
5. Share the resulting URL with your team.

## Workflow
Use the tabs in order, 1 → 10. Each tab depends on the previous results.
