# Push SENTINEL to GitHub (github.com/ankurgenomics)

This repo is **standalone** — it contains only the SENTINEL project (no SFA decks,
resume, or career-ops files). Git is initialised inside this `sentinel-amr/` folder,
so nothing else gets included.

## 0. Revoke the exposed token first (important)

Your `career-ops` git remote had a personal access token embedded in its URL
(`ghp_...`). Treat it as **compromised**:

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Revoke that token
3. Generate a fresh one only if you need token-based auth (a fine-grained token
   scoped to a single repo is safest)

This SENTINEL repo never contains that token — verified before commit.

## 1. Create the empty repo on GitHub

Option A — web UI (simplest):
- Go to https://github.com/new
- Owner: **ankurgenomics**, name: **sentinel-amr** (or `SENTINEL`)
- Public, **do NOT** add README/LICENSE/.gitignore (we already have them)
- Create repository

Option B — GitHub CLI (if you re-auth `gh auth login` first):
```bash
gh repo create ankurgenomics/sentinel-amr --public --source=. --remote=origin --push
```
(If you use Option B, you can skip steps 2–3.)

## 2. Point the local repo at your new remote

From inside this folder (`sentinel-amr/`):

```bash
git remote add origin https://github.com/ankurgenomics/sentinel-amr.git
```

If a remote named `origin` already exists, replace it:
```bash
git remote set-url origin https://github.com/ankurgenomics/sentinel-amr.git
```

## 3. Push

```bash
git push -u origin main
```

You'll be prompted for credentials. Use either:
- **GitHub CLI / Git Credential Manager** (recommended — no token in the URL), or
- A **fresh fine-grained PAT** as the password when prompted.

Do **not** paste a token into the remote URL.

## 4. Verify on GitHub

- README renders with the architecture diagram and honest results table
- `models/`, `output/`, `metrics/`, `.venv/` are **absent** (they're gitignored
  and regenerate via `./run.sh`)
- `data/sentinel_dataset.csv` **is** present (small, for the offline demo)
- Actions tab: none configured (add CI later if you want)

## What a reviewer does after cloning

```bash
git clone https://github.com/ankurgenomics/sentinel-amr.git
cd sentinel-amr
./run.sh            # venv + deps + train + explain + novelty + demo + how-to
# or just the tests:
.venv/bin/python -m pytest -q
```
