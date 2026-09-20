"""
Import app.py against a stubbed Streamlit.

Streamlit scripts execute top to bottom, so importing the module with every
widget returning a falsy default exercises every code path that runs before a
user clicks anything — which is where signature errors and typos live. It
cannot test rendering, but it catches "that function does not take that
argument" before deploy.
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _Ctx:
    """Anything that can be used as a context manager, iterated or called."""

    def __init__(self, n=4):
        self._n = n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([_Ctx() for _ in range(self._n)])

    def __getitem__(self, i):
        return _Ctx()

    def __getattr__(self, name):
        return _Widget(name)

    def __call__(self, *a, **k):
        return _Ctx()


class _Widget:
    """A callable that returns a falsy, duck-typed default."""

    FALSY = {
        "button", "checkbox", "toggle", "form_submit_button", "download_button",
    }
    # Streamlit returns None for an empty uploader, or [] with multiple files.
    NONE = {"file_uploader", "camera_input"}
    TEXT = {"text_input", "text_area", "selectbox", "radio"}
    LIST = {"multiselect"}
    NUM = {"slider", "number_input"}

    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        n = self.name
        if n == "columns":
            spec = a[0] if a else 2
            count = spec if isinstance(spec, int) else len(spec)
            return [_Ctx() for _ in range(count)]
        if n == "tabs":
            return [_Ctx() for _ in range(len(a[0]) if a else 3)]
        if n in self.NONE:
            return [] if k.get("accept_multiple_files") else None
        if n in self.FALSY:
            return False
        if n in self.TEXT:
            if n == "selectbox":
                opts = k.get("options") or (a[1] if len(a) > 1 else [""])
                return opts[k.get("index", 0)] if opts else ""
            return ""
        if n in self.LIST:
            return []
        if n in self.NUM:
            return a[3] if len(a) > 3 else 2
        return _Ctx()


class _SessionState(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Secrets(dict):
    def get(self, k, default=None):
        return default


def build_stub() -> types.ModuleType:
    st = types.ModuleType("streamlit")
    st.session_state = _SessionState()
    st.secrets = _Secrets()
    st.sidebar = _Ctx()

    def _getattr(name):
        return _Widget(name)

    st.__getattr__ = _getattr
    for name in ("set_page_config", "title", "caption", "header", "subheader",
                 "markdown", "code", "text", "divider", "success", "info",
                 "error", "warning", "metric", "progress", "spinner",
                 "expander", "container", "columns", "tabs", "button",
                 "checkbox", "text_input", "text_area", "selectbox", "slider",
                 "multiselect", "file_uploader", "download_button", "empty",
                 "write", "json", "dataframe", "toggle", "radio", "number_input",
                 "form", "form_submit_button", "image", "stop", "rerun"):
        setattr(st, name, _Widget(name))
    return st


def main() -> int:
    sys.modules["streamlit"] = build_stub()
    try:
        import importlib
        app = importlib.import_module("app")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nFAIL — app.py raised {type(exc).__name__}: {exc}")
        return 1

    print("PASS — app.py imported and executed its no-input path cleanly.")

    # the helpers app.py defines should exist with the shapes the UI expects
    for fn in ("client_for", "_final_tex", "_score_card"):
        if not hasattr(app, fn):
            print(f"FAIL — app.py is missing {fn}")
            return 1
    print("PASS — UI helpers present.")

    # exercise _final_tex for real
    from core.agent import Draft
    from core.latexdoc import parse
    doc = parse(open(os.path.join(ROOT, "samples", "aalekh_resume.tex")).read())
    b = doc.bullets[0]
    draft = Draft(round_no=1, bullet_edits={b.bid: "Rewritten first bullet for the test."},
                  skill_edits={})
    kept = app._final_tex(doc, draft, {b.bid: True})
    dropped = app._final_tex(doc, draft, {b.bid: False})
    ok = ("Rewritten first bullet" in kept
          and "Rewritten first bullet" not in dropped
          and dropped == doc.source)
    print(("PASS" if ok else "FAIL") + " — per-bullet accept/reject rebuilds the file correctly.")
    if not ok:
        return 1

    # ---- second pass: re-run the script with a full result loaded ----------
    # This is where the real rendering code lives: gap rows, diffs, download
    # buttons, parseability. None of it runs on an empty session.
    print("\n-- second pass: populated session --")
    import json as _json
    from core.agent import tailor
    from core.jd_extract import heuristic_analysis
    from core.jd_fetch import JobPosting
    from core.llm import StubLLM
    from core.matcher import build_report
    from core.ats import score as ats_score
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_pipeline as tp

    jd_text = open(os.path.join(ROOT, "samples", "jd_google_backend.txt")).read()
    jd = heuristic_analysis(jd_text, "Senior Software Engineer, Backend Infrastructure", "Google")
    report = build_report(doc, jd, ["Kafka", "Event Driven"])
    res = tailor(doc, jd, StubLLM(handlers=tp.STUB),
                 declared_skills=["Kafka", "Event Driven"],
                 context_notes="The EFE services run on Java 21.", max_rounds=2)

    st = sys.modules["streamlit"]
    st.session_state.clear()
    st.session_state.update({
        "doc": doc, "tex_name": "aalekh_resume.tex", "cls_files": {},
        "job": JobPosting(text=jd_text, ok=True, title="Senior Software Engineer",
                          company="Google", source="pasted"),
        "jd": jd, "report": report, "ats_before": ats_score(doc, jd, report),
        "result": res,
        "accepted": {c.bid: True for c in res.best.changes if c.changed},
        "declared": ["Kafka"], "context_notes": "", "api_key": "",
        "pdf_after": None, "cover": None, "outreach": None, "prep": None, "plan": None,
    })
    os.environ["ANTHROPIC_API_KEY"] = "stub-key-for-smoke-test"
    try:
        importlib.reload(app)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FAIL — populated render raised {type(exc).__name__}: {exc}")
        return 1
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    print("PASS — every tab rendered with a full result in session state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
