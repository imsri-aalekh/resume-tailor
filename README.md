# Resume Tailor

Takes your LaTeX resume and one job posting, and rewrites your bullets to match
the posting — without touching your template and without inventing anything.

Built around one idea: **your resume is the source of truth.** The agent may
re-frame what you already did. It may not make things up. That rule is enforced
in code, not in a prompt, and every blocked rewrite is shown to you.

---

## What it actually does

**1. Reads your `.tex` without regenerating it.**
The parser records the exact character offsets of your bullet text and skill
values. Rewriting replaces only those spans. Your `\documentclass`, preamble,
macros, spacing and section scaffolding come through byte-for-byte identical —
the app verifies this on every run and refuses to proceed if it cannot.

**2. Reads the posting.**
URL or paste. LinkedIn, Naukri, Greenhouse and Lever get dedicated handling;
anything emitting schema.org `JobPosting` works automatically. When a site
blocks the server it says so plainly instead of feeding you a login page as if
it were a job description.

**3. Shows you the gap before it changes anything.**
Every requirement is graded on *evidence*, not presence:

| | meaning |
|---|---|
| ✅ Strong | proven in a bullet, with a number or a strong verb |
| 🟢 Present | mentioned, but thin |
| 🟡 Listed only | in your skills line with no story behind it |
| 🔵 You know it | you ticked it; not on the resume yet |
| 🔴 Missing | no evidence anywhere |

The 🟡 row is the one that matters. "Apache Kafka" in a skills list gets you
past a filter and then falls apart in the phone screen.

**4. Tailors, criticises itself, and revises.**
A writer drafts. A **separate critic** — briefed as a hiring manager with 90
seconds and an interviewer who will deep-dive every claim — attacks the draft
looking for fabrication, keyword stuffing, and lost substance. A reviser fixes
what it found. Best draft wins on a composite of measured ATS gain and critic
score, with fabrication flags heavily penalised.

**5. Refuses to lie on your behalf.**
Before any draft is scored, `core/guards.py` rejects:

- **invented numbers** — every figure must already exist in your resume or in
  the extra-context box you filled in yourself
- **unbacked skills** — anything outside your resume + what you explicitly
  confirmed
- **entity drift** — silently dropping the company, vendor or system name that
  made the bullet credible
- **broken LaTeX** — unbalanced braces, unescaped `%` and `&`, stray macros

Rejected rewrites revert to your original text and are handed back to the critic
as violations it has to resolve, so the loop cannot converge on a lie.

**6. Gives you bullet-level control.**
Every change shows an inline word diff, the reason, and which part of the
original supports it. Untick anything; the download follows your choices.

---

## Beyond the resume

- **Cover note** — under 200 words, built from your strongest matching bullets.
- **Recruiter and hiring-manager messages** — under 90 words, no buzzwords.
- **Interview risk** — every bullet you sharpened is a question you have
  invited. This lists the questions and flags anything thin.
- **Gap plan** — for what no rewrite can fix: the smallest project that would
  earn one honest bullet, and how to bridge the gap if asked before then.

---

## Deploying to Streamlit Community Cloud

1. **Push this folder to a GitHub repo.**
   ```bash
   cd resume-tailor
   git init && git add . && git commit -m "Resume Tailor"
   gh repo create resume-tailor --private --source=. --push
   ```

2. **Create the app.** Go to <https://share.streamlit.io> → *Create app* → pick
   the repo → main file `app.py` → Deploy.

3. **Add your key.** App settings → **Secrets** → paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Save. The app restarts and picks it up. You can also paste a key into the
   sidebar for a one-off session without storing it.

4. **Upload your `.cls`** in the sidebar the first time, so the PDF preview
   matches your real template.

That is the whole deployment. No build steps, no Dockerfile.

### About PDF preview on Streamlit Cloud

There is no LaTeX toolchain on Streamlit Cloud, and installing one via
`packages.txt` is about 1.5 GB — it usually times out the build and leaves you
with a broken app rather than a slow one. So PDF preview is **off there by
design**, and the app tells you so instead of failing quietly.

This costs you nothing that matters. The `.tex` download is the real deliverable
and it uses your template unchanged, so Overleaf compiles it exactly as your
current resume compiles. If you do want server-side PDF, rename
`packages.txt.optional` to `packages.txt` and accept the build risk, or run
locally where you already have LaTeX.

---

## Running locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

With a local LaTeX install (`brew install --cask mactex-no-gui`, or
`apt install texlive-latex-extra texlive-xetex latexmk`) the PDF preview and the
real ATS parseability test both switch on automatically.

---

## Tests

```bash
python3 tests/test_pipeline.py    # full agent loop, offline, no API key
python3 tests/test_app_smoke.py   # imports app.py against a stubbed Streamlit
```

`test_pipeline.py` runs the whole loop with a stub model that *deliberately
misbehaves* in round one — it invents a 63% latency figure and claims Kubernetes
— and asserts that the guards catch both, that the originals are restored, and
that the clean second draft wins.

---

## Layout

```
app.py                  Streamlit UI — five tabs, no logic of its own
core/
  latexdoc.py           span-preserving .tex parser; render(doc,{}) == source
  jd_fetch.py           posting retrieval with per-site strategies
  jd_extract.py         JD → structured requirements (LLM + offline fallback)
  skills.py             alias table, evidence grading, adjacency
  matcher.py            gap report, coverage, allowed-vocabulary boundary
  ats.py                relevance scoring + real parseability test on the PDF
  guards.py             the rules the agent cannot talk its way around
  agent.py              write → validate → critique → revise loop
  extras.py             cover note, outreach, interview risk, gap plan
  render.py             PDF compilation with graceful degradation
  diffing.py            word-level diffs
samples/                a resume and a job posting to try it on
tests/
```

---

## Honest limits

- **LinkedIn and Naukri block datacenter IPs.** The dedicated endpoints work
  more often than plain scraping, but pasting the text is the reliable path.
- **The ATS score is a diagnostic, not a target.** Above ~85 usually means the
  resume is being written for the robot rather than the human. The app warns you
  when keyword density gets there.
- **A tailored resume cannot fix a real gap.** When something is genuinely
  missing the app says so and declines to paper over it. That is the point.
