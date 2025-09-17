# Coffee Sales Forecaster

Forecast **daily revenue** from `data/Coffe_sales.csv` using Prophet.  
No local setup required — use **Colab** for quick testing and **Streamlit Cloud** for hosting.

## 1) Data
Place your CSV at:
data/Coffe_sales.csv

## 2) Run in Colab
- Open a new Colab notebook.
- Copy blocks A–F from the instructions.
- Update `GITHUB_RAW_URL` to your repo path if desired.

## 3) Deploy to Streamlit
- Push `app.py`, `requirements.txt`, and your `data/` folder to GitHub.
- Go to streamlit.io → Deploy App → point to this repo → `app.py`.
- Optional: edit the default GitHub raw URL at the top of the app, or just upload CSV in the UI.

## Notes
- The app aggregates transaction rows to **daily revenue**.
- Forecast horizon selectable (7–90 days).
- No model files are saved.