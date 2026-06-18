# 🚀 Deploying ParkSensei to Streamlit Community Cloud

This gives you a **free, public Demo Link** (e.g. `https://ParkSensei.streamlit.app`) — exactly what the
Round-2 form's **Demo Link** field needs. Anyone can open it; no install.

---

## What you need (both free)
- A **GitHub** account
- A **Streamlit Community Cloud** account → sign in with GitHub at **https://share.streamlit.io**

---

## The repo is already deploy-ready
This `round2/` folder has everything Streamlit Cloud needs:

| File | Purpose |
|---|---|
| `app/app.py` | entry point (the main file Cloud runs) |
| `requirements.txt` | pinned dependencies |
| `.streamlit/config.toml` | dark theme |
| `data/clean.pkl`, `data/junctions.pkl` | **prebuilt data (~32 MB), committed** so the app runs without the raw CSV |
| `.gitignore` | excludes the 109 MB raw datasets (GitHub rejects >100 MB) and scratch files |

> The raw `dataset theme 1/*.csv` is intentionally **not** committed — the app only needs the pickles.
> To regenerate them locally: `python app/prep.py` (with the CSV in place).

---

## Step 1 — Push `round2/` to a new GitHub repo
Open a terminal **in the `round2/` folder** and run:

```bash
git init
git add .
git commit -m "ParkSensei - Gridlock 2.0 Round 2 prototype"
git branch -M main
# 1) create an empty repo named e.g. "ParkSensei" on github.com (no README), then:
git remote add origin https://github.com/<your-username>/ParkSensei.git
git push -u origin main
```

✅ **Sanity check on GitHub:** `data/clean.pkl` and `data/junctions.pkl` are present, and there is **no**
`dataset theme 1/` folder.

---

## Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io** → sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Set:
   - **Repository:** `<your-username>/ParkSensei`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. Open **Advanced settings** → set **Python version = 3.12**
   *(important — our pinned pandas/numpy ship wheels for 3.12; a newer Python can fail the build).* No
   secrets needed.
5. Click **Deploy**. First build takes ~2–5 minutes.

---

## Step 3 — Use the link
You'll get a public URL like `https://<name>.streamlit.app`. Paste it into the Round-2 form's
**Demo Link** field. Done.

To update the live app later, just `git push` to `main` — Cloud auto-redeploys.

---

## Optional — enable the "Ask ParkSensei" AI co-pilot
The co-pilot page needs a free **Google Gemini API key** (get one at `aistudio.google.com/apikey`). The rest
of the app runs fine without it (the page just shows a setup card).
- **Local:** create `.streamlit/secrets.toml` with:
  ```toml
  GEMINI_API_KEY = "AIza..."
  # COPILOT_MODEL = "gemini-2.5-flash"   # optional override (e.g. gemini-2.5-pro)
  ```
- **Streamlit Cloud:** *Manage app → Settings → Secrets* → paste the same line → reload.

> `.streamlit/secrets.toml` is git-ignored — **never commit your key.**

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| Build fails on `pandas`/`numpy` | You didn't pick **Python 3.12** — Manage app → Settings → set 3.12 → Reboot |
| "Oh no" / out-of-memory | Unlikely (data is slimmed to ~153 MB RAM, cap is 1 GB). If it happens, downcast text cols to `category` and re-push |
| Map tiles blank | The dark basemap loads from CartoCDN at runtime — just needs internet (Cloud has it) |
| Push rejected (file too large) | A `dataset theme*/` folder slipped in — confirm `.gitignore` is committed and re-`git add` |

---

## Don't want to deploy right now?
You can screen-record the app locally instead (`streamlit run app/app.py`). But a live public link scores
higher on "working demo" and lets judges click around themselves — recommended.
