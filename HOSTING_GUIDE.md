# Hosting Guide (No Docker)

This is the quickest way to demo the project for recruiters: host the API on Render and the Streamlit UI on Streamlit Community Cloud.

## 1) FastAPI API on Render
- Connect your GitHub repo.
- Runtime: Python 3.11
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Port setting: 8000
- Optional env vars: `PYTHONUNBUFFERED=1`; add any data/model paths if you externalize data.
- After deploy, copy the public URL (e.g., `https://ev-api.onrender.com`).

## 2) Streamlit UI on Streamlit Community Cloud
- App file: `streamlit_app/Home.py`
- Python version: 3.11
- Requirements file: repo root `requirements.txt`
- Secrets: set `API_BASE_URL` to your Render API URL. Add SMTP creds if you plan to email forecasts (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`).
- If the UI loads local CSVs, keep `data/processed` in the repo. For large/private data, move to a hosted source and update loaders.

## 3) Local run (verify before sharing)
- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- UI: `streamlit run streamlit_app/Home.py --server.address 0.0.0.0 --server.port 8501`
- Point UI to local API with env/secrets: `API_BASE_URL=http://localhost:8000`

## 4) What to share with recruiters
- UI link (Streamlit): `https://your-app.streamlit.app`
- API docs link (Render): `https://ev-api.onrender.com/docs`
- Quick local commands (for README): see section 3 above.

## 5) Troubleshooting
- UI cannot reach API: confirm `API_BASE_URL` is set to the live API URL; CORS is already open on the API.
- Data not found: ensure `data/processed/cleaned_ev_sessions.csv` exists in the deployed repo, or switch loaders to a hosted data source.
- Prophet install slow: stays in requirements; if builds time out, pin a lighter version or prebuild wheels.

## Optional: wire API_BASE_URL in code
Pages currently assume local data; if you add API calls, read `API_BASE_URL` from `st.secrets` or `os.environ` with a sensible default (e.g., fallback to `http://localhost:8000`).
