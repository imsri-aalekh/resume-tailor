"""
Check this repo is ready for Streamlit Community Cloud before you push.

A failed cloud build takes several minutes to tell you something a one-second
check here would have. Run it after any dependency or layout change:

    python3 preflight.py
"""

from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROBLEMS: list[str] = []
NOTES: list[str] = []

# Directories that are never part of what gets deployed. Kept in one place:
# these walks used to disagree, and the import scan happily parsed every .py
# inside .venv — thousands of files, and third-party imports reported as
# "missing from requirements.txt".
# Only bulk directories that are never deployed. Deliberately NOT .streamlit:
# that is exactly where a secrets.toml with a live key would sit, and the leak
# scan below has to be able to see it.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages"}


def fail(msg: str) -> None:
    PROBLEMS.append(msg)
    print(f"  ✗ {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def note(msg: str) -> None:
    NOTES.append(msg)
    print(f"  · {msg}")


print("Streamlit Cloud preflight\n")

# ---------------------------------------------------------------- layout ----
print("Layout")
if os.path.exists(os.path.join(ROOT, "app.py")):
    ok("app.py is at the repo root (this is the 'Main file path' on deploy)")
else:
    fail("app.py missing from the repo root")

if os.path.exists(os.path.join(ROOT, "requirements.txt")):
    ok("requirements.txt present")
else:
    fail("requirements.txt missing — the build will install nothing")

if os.path.exists(os.path.join(ROOT, "packages.txt")):
    note("packages.txt exists — apt packages will be installed. If this is "
         "texlive, expect a long or failed build.")

# --------------------------------------------------------------- secrets ----
print("\nSecrets hygiene")
secret_path = os.path.join(ROOT, ".streamlit", "secrets.toml")
gitignore = ""
gi_path = os.path.join(ROOT, ".gitignore")
if os.path.exists(gi_path):
    gitignore = open(gi_path).read()

if os.path.exists(secret_path):
    if "secrets.toml" in gitignore:
        ok("secrets.toml exists locally but is gitignored")
    else:
        fail("secrets.toml exists and is NOT gitignored — you would publish your key")
else:
    ok("no secrets.toml in the tree")

# scan tracked files for anything that looks like a live key
key_re = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")
leaks = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn.endswith((".png", ".pdf", ".zip", ".jpg")):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            body = open(fp, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if key_re.search(body):
            leaks.append(os.path.relpath(fp, ROOT))
if leaks:
    fail(f"API key pattern found in: {', '.join(leaks)}")
else:
    ok("no API key patterns anywhere in the tree")

# ---------------------------------------------------------- dependencies ----
print("\nDependencies")
reqs = []
if os.path.exists(os.path.join(ROOT, "requirements.txt")):
    for line in open(os.path.join(ROOT, "requirements.txt")):
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
            if line.startswith(("-e", "file:", "/", ".")) or "@" in line:
                fail(f"local or VCS requirement will not install on Cloud: {line}")
ok(f"{len(reqs)} requirement(s): {', '.join(r.split('>')[0].split('=')[0] for r in reqs)}")

# every third-party module app.py and core/ import must be declared
declared = {re.split(r"[<>=!\[]", r)[0].strip().lower().replace("-", "_") for r in reqs}
ALIASES = {"beautifulsoup4": "bs4", "pillow": "pil"}
declared |= {ALIASES[d] for d in list(declared) if d in ALIASES}
STDLIB = set(sys.stdlib_module_names)
LOCAL = {"core", "app"}

missing = set()
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS | {"tests"}]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            tree = ast.parse(open(fp, encoding="utf-8").read())
        except SyntaxError as exc:
            fail(f"{os.path.relpath(fp, ROOT)} does not parse: {exc}")
            continue
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                for a in node.names:
                    mod = a.name.split(".")[0]
                    if mod not in STDLIB and mod not in LOCAL and mod not in declared:
                        missing.add(f"{mod} (in {os.path.relpath(fp, ROOT)})")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module.split(".")[0]
                if mod not in STDLIB and mod not in LOCAL and mod not in declared:
                    missing.add(f"{mod} (in {os.path.relpath(fp, ROOT)})")
if missing:
    for m in sorted(missing):
        fail(f"imported but not in requirements.txt: {m}")
else:
    ok("every third-party import is declared in requirements.txt")

# ------------------------------------------------------------ file sizes ----
print("\nRepo size")
big = []
total = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        fp = os.path.join(dirpath, fn)
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        total += size
        if size > 20 * 1024 * 1024:
            big.append(f"{os.path.relpath(fp, ROOT)} ({size/1e6:.1f} MB)")
if big:
    fail("files over 20 MB: " + ", ".join(big))
else:
    ok(f"repo is {total/1024:.0f} KB — well inside limits")

# --------------------------------------------------------------- imports ----
print("\nImportability")
sys.path.insert(0, ROOT)
for mod in ("core.latexdoc", "core.skills", "core.matcher", "core.ats",
            "core.guards", "core.agent", "core.jd_fetch", "core.jd_extract",
            "core.llm", "core.render", "core.diffing", "core.extras"):
    try:
        __import__(mod)
    except Exception as exc:
        fail(f"{mod} fails to import: {type(exc).__name__}: {exc}")
else:
    if not PROBLEMS or all("fails to import" not in p for p in PROBLEMS):
        ok("all core modules import with no optional dependencies")

try:
    import streamlit  # noqa: F401
    ok("streamlit importable locally")
except ImportError:
    note("streamlit not installed here — fine, Cloud installs it from requirements.txt")

# -------------------------------------------------------------- provider ----
print("\nModel provider")
try:
    from core.llm import PROVIDERS, DEFAULT_PROVIDER
    _p = PROVIDERS[DEFAULT_PROVIDER]
    if _p.local:
        _hosted = [k for k, v in PROVIDERS.items() if not v.local]
        note(f"default provider is '{DEFAULT_PROVIDER}' — local to this "
             f"machine. On Streamlit Cloud nothing listens there, so the "
             f"sidebar falls back to whichever hosted provider has a key in "
             f"secrets ({', '.join(_hosted)}). Put at least one key in Cloud "
             f"secrets or the deployed app opens with no working provider; "
             f"GROQ_API_KEY is the free one. Local runs are unaffected.")
    else:
        ok(f"default provider '{DEFAULT_PROVIDER}' is hosted — deployable as-is")
    for _k, _pr in PROVIDERS.items():
        if _pr.free and not _pr.local and not _pr.rpm:
            fail(f"free provider '{_k}' declares no rpm — requests cannot be paced")
    ok("every metered free provider declares a per-minute budget")
except Exception as exc:                      # noqa: BLE001
    fail(f"provider table did not load: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- python ----
print("\nRuntime")
major, minor = sys.version_info[:2]
ok(f"checked under Python {major}.{minor}")
note("On Streamlit Cloud pick Python 3.11 or 3.12 in Advanced settings — "
     "3.13 wheels for lxml/pdfplumber are newer and occasionally lag.")

# ----------------------------------------------------------------- verdict --
print("\n" + "=" * 58)
if PROBLEMS:
    print(f"NOT READY — {len(PROBLEMS)} problem(s) to fix first.")
    sys.exit(1)
print("READY TO DEPLOY")
if NOTES:
    print("\nWorth knowing:")
    for n in NOTES:
        print(f"  · {n}")
sys.exit(0)
