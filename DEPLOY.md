# Deploy

Two steps: GitHub, then Streamlit. About four minutes end to end.

The repo is already initialised with one commit and a clean `.gitignore`, so
nothing here creates the repo from scratch — it just pushes what exists.

---

## Step 1 — GitHub

### If you have the `gh` CLI

```bash
cd resume-tailor
gh auth login          # skip if already authenticated
gh repo create resume-tailor --private --source=. --remote=origin --push
```

### If you don't (plain git + a token)

Create an empty repo at <https://github.com/new> — name it `resume-tailor`,
set it **private**, and do **not** add a README, .gitignore or licence (the repo
already has them, and an initial commit on their side forces a merge).

Then:

```bash
cd resume-tailor
git remote add origin https://github.com/<your-username>/resume-tailor.git
git push -u origin main
```

Git will ask for a username and password. The password is a **personal access
token**, not your account password — make one at
<https://github.com/settings/tokens> with `repo` scope.

### Verify

```bash
git ls-remote origin          # should list refs/heads/main
```

---

## Step 2 — Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with the **same GitHub
   account**. Authorise it for private repos when prompted — otherwise your
   repo will not appear in the list.

2. **Create app** → **Deploy a public app from GitHub**.

3. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `<your-username>/resume-tailor` |
   | Branch | `main` |
   | Main file path | `app.py` |

4. Open **Advanced settings** before deploying:

   - **Python version** → `3.11` or `3.12`. Not 3.13 — the `lxml` and
     `pdfplumber` wheels lag a release behind and the build can fail on it.
   - **Secrets** → paste exactly this, with your real key:

     ```toml
     ANTHROPIC_API_KEY = "sk-ant-..."
     ```

5. **Deploy**. First build takes 2–4 minutes (it is compiling nothing; it is
   just pip and the container).

The app reads the key from secrets automatically — the sidebar will say "API key
loaded from secrets" rather than showing a key box.

---

## Right after it comes up

1. **Inputs tab** → upload your real `.tex`. Check the green box says it
   reproduced your file byte-for-byte, and that the bullet count matches what
   you actually have. If the count is wrong, send me the `.tex` — the parser
   needs a case added.
2. **Sidebar** → upload `resume-openfont.cls` so the PDF preview is exact.
   (PDF preview only works if you enabled the optional `packages.txt`; by
   default it is off on Cloud and the `.tex` download is the deliverable.)
3. Paste a real job posting from your target list and run the gap analysis.

---

## Changing the app later

Streamlit Cloud redeploys on every push to `main`:

```bash
git add -A && git commit -m "…" && git push
```

Before pushing, run:

```bash
python3 preflight.py    # catches what would break the Cloud build
./run_tests.sh          # full offline suite, no API key needed
```

---

## If the build fails

Click **Manage app** in the bottom right for the build log.

| Log says | Fix |
|---|---|
| `No matching distribution found for lxml` | Python version is 3.13 — change it in Advanced settings and reboot the app |
| `ModuleNotFoundError: No module named 'core'` | Main file path is wrong; it must be `app.py` at the repo root, not a path into a subfolder |
| `KeyError: 'ANTHROPIC_API_KEY'` | Secrets were not saved, or were pasted as `ANTHROPIC_API_KEY=...` without quotes. TOML needs the quotes |
| Build hangs, then times out | You renamed `packages.txt.optional` to `packages.txt`. Rename it back — texlive does not fit |
| Repo not listed when creating the app | Streamlit was not authorised for private repos. Revoke and re-authorise, or make the repo public |
