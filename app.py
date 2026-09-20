"""
Resume Tailor — Streamlit front end.

Two inputs, as asked: your resume and a job posting. Everything else is either
a safety net or a way to stay in control of what gets changed.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

from core import extras, skills as S
from core.agent import Draft, TailorResult, tailor
from core.ats import (ATSScore, WEIGHTS, parseability_from_pdf,
                      score as ats_score)
from core.diffing import change_stats, to_html, unified
from core.jd_extract import JDAnalysis, analyse
from core.jd_fetch import JobPosting, fetch
from core.latexdoc import ResumeDoc, parse as parse_tex, render
from core.llm import (PROVIDERS, ENV_VARS, LLMClient, SMART_MODEL,
                      DEFAULT_PROVIDER, installed_models, local_server_up)
from core.matcher import GapReport, LEVEL_LABEL, build_report
from core.render import available_engines, compile_tex, shim_class

st.set_page_config(page_title="Resume Tailor", page_icon="📄", layout="wide")

LEVEL_COLOR = {
    "strong": "#1f7a3d",
    "present": "#4a7a1f",
    "listed_only": "#8a6d1f",
    "declared": "#1f5a8a",
    "missing": "#8a2f2f",
}
LEVEL_ICON = {
    "strong": "✅", "present": "🟢", "listed_only": "🟡",
    "declared": "🔵", "missing": "🔴",
}

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  .pill {display:inline-block;padding:2px 9px;border-radius:11px;
         font-size:0.74rem;font-weight:600;color:#fff;margin-right:6px;}
  .gaprow {padding:9px 12px;border-radius:8px;margin-bottom:6px;
           border-left:4px solid #555;background:rgba(128,128,128,0.08);}
  .metricbig {font-size:2.5rem;font-weight:700;line-height:1;}
  .muted {color:#888;font-size:0.84rem;}
  .bulletbox {border:1px solid rgba(128,128,128,0.25);border-radius:8px;
              padding:12px 14px;margin-bottom:10px;}
  code {font-size:0.82rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------

def _init() -> None:
    defaults = {
        "doc": None, "tex_name": "", "cls_files": {}, "job": None, "jd": None,
        "report": None, "ats_before": None, "result": None, "accepted": {},
        "pdf_before": None, "pdf_after": None, "history": [],
        "cover": None, "outreach": None, "prep": None, "plan": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init()


def secret(name: str) -> str:
    """Read a key from the sidebar, then the environment, then st.secrets."""
    val = st.session_state.get("api_key") or ""
    if val:
        return val
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        return st.secrets.get(name, "") or ""
    except Exception:
        return ""


def _initial_provider() -> str:
    """Which provider the sidebar should open on.

    Local-first when a local server is actually there — that is the best
    experience and costs nothing. But the same code is deployed to Streamlit
    Cloud, where localhost:11434 is the container itself and nothing is
    listening. Opening on a dead provider there would make every visitor
    diagnose and switch by hand, so fall back to whichever hosted provider
    already has a key in secrets.
    """
    # No st.cache_data: local_server_up already caches its probe for a few
    # seconds, the secret lookups are dict reads, and a Streamlit cache here
    # would pin a stale answer when you start Ollama mid-session.
    if local_server_up(DEFAULT_PROVIDER):
        return DEFAULT_PROVIDER
    for key, var in ENV_VARS.items():
        if not var:
            continue
        if os.environ.get(var):
            return key
        try:
            if st.secrets.get(var, ""):
                return key
        except Exception:
            pass
    return DEFAULT_PROVIDER


def client_for(model: str) -> LLMClient:
    prov = st.session_state.get("provider", DEFAULT_PROVIDER)
    env_name = ENV_VARS.get(prov, "ANTHROPIC_API_KEY")
    key = secret(env_name) if env_name else ""
    return LLMClient(api_key=key, model=model, provider=prov)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    prov_keys = list(PROVIDERS)
    prov_choice = st.selectbox(
        "Model provider", prov_keys,
        format_func=lambda k: PROVIDERS[k].label,
        index=prov_keys.index(_pick if (_pick := _initial_provider()) in prov_keys
                              else DEFAULT_PROVIDER),
        help="Gap analysis and ATS scoring need no provider at all. Only the "
             "tailoring agent and the writing extras call a model.",
    )
    st.session_state["provider"] = prov_choice
    prov = PROVIDERS[prov_choice]
    st.caption(prov.note)

    env_name = ENV_VARS.get(prov_choice, "")

    local_models: list = []
    if prov.local:
        # No key to collect. What matters instead is whether the server is up
        # and which models are actually pulled.
        st.session_state["api_key"] = ""
        local_models = installed_models(prov_choice)
        if local_models:
            st.success(f"Ollama is running · {len(local_models)} model(s) local.")
        else:
            st.warning("Ollama is not responding on localhost:11434.")
            st.code("ollama serve\nollama pull " + prov.default_model, language="bash")
    else:
        have_stored = False
        try:
            have_stored = bool(st.secrets.get(env_name, ""))
        except Exception:
            have_stored = False
        have_stored = have_stored or bool(os.environ.get(env_name))

        if have_stored:
            st.success(f"`{env_name}` loaded from secrets.")
            st.session_state["api_key"] = ""
        else:
            st.session_state["api_key"] = st.text_input(
                f"{prov.label} API key", type="password",
                help=f"Kept only for this browser session. For a permanent setup "
                     f"put {env_name} in Streamlit secrets.",
            )
            st.caption(f"Get a key → {prov.console}")

    if prov.local:
        # Offer what is actually pulled; the default first if it is there.
        model_options = (sorted(local_models,
                                key=lambda m: m != prov.default_model)
                         or [prov.default_model])
    elif prov_choice == "anthropic":
        model_options = [prov.default_model, SMART_MODEL]
    else:
        # Provider model names churn — Groq decommissioned its free Llama
        # outright — so offer the alternates rather than making people guess.
        model_options = [prov.default_model, *prov.alt_models]
    model = st.selectbox("Model", model_options, index=0)
    model = st.text_input(
        "…or type a model name", value=model,
        help="Provider model names change often. If the app reports an unknown "
             "model, copy the current name from the provider's docs.",
    ) or model

    rounds = st.slider("Critique rounds", 1, 4, 2,
                       help="Each round is a full critique and revision pass. "
                            "Two is usually where it stops finding real problems. "
                            "Drop to 1 if a free tier is rate-limiting you.")

    if prov.local:
        st.info(
            "Small local models follow the strict rewrite rules less reliably "
            "than a frontier model, so expect more blocked rewrites. That is "
            "the safety net doing its job — a blocked rewrite keeps your "
            "original wording, never a fabricated one. A larger local model, "
            "or more RAM, buys back most of the difference."
        )
    elif prov.free:
        st.info(
            "Free-tier models follow the strict rewrite rules less reliably, so "
            "expect more blocked rewrites. That is the safety net doing its job "
            "— a blocked rewrite keeps your original wording."
        )

    st.divider()
    st.subheader("Template class file")
    st.caption(
        "Your resume uses a custom .cls. Upload it here to get an exact PDF "
        "preview. Without it the preview uses a stand-in and the spacing will "
        "differ — the .tex you download is unaffected either way."
    )
    cls_up = st.file_uploader("resume-openfont.cls (or your class file)",
                              type=["cls", "sty"], accept_multiple_files=True)
    if cls_up:
        st.session_state["cls_files"] = {f.name: f.getvalue() for f in cls_up}
        st.success(f"Loaded {', '.join(st.session_state['cls_files'])}")

    st.divider()
    engines = available_engines()
    if engines:
        st.caption(f"PDF engine: `{engines[0]}`")
    else:
        st.caption("No LaTeX engine on this server — PDF preview is off. "
                   "The .tex download and Overleaf hand-off still work.")


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------

st.title("Resume Tailor")
st.caption(
    "Rewrites your existing resume against one job posting. Your LaTeX template "
    "is never regenerated — only the text inside your bullets changes."
)

tab_input, tab_gaps, tab_tailor, tab_output, tab_next = st.tabs(
    ["1 · Inputs", "2 · Gap analysis", "3 · Tailor", "4 · Download", "5 · After you apply"]
)


# --------------------------------------------------------------------------
# 1. inputs
# --------------------------------------------------------------------------

with tab_input:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Your resume")
        up = st.file_uploader("LaTeX source (.tex)", type=["tex", "txt"])
        use_sample = st.checkbox("Use the bundled sample resume instead")

        tex_src = None
        if up is not None:
            tex_src = up.getvalue().decode("utf-8", errors="replace")
            st.session_state["tex_name"] = up.name
        elif use_sample:
            path = os.path.join(os.path.dirname(__file__), "samples", "aalekh_resume.tex")
            if os.path.exists(path):
                tex_src = open(path, encoding="utf-8").read()
                st.session_state["tex_name"] = "aalekh_resume.tex"

        if tex_src:
            doc = parse_tex(tex_src)
            st.session_state["doc"] = doc
            ok = render(doc, {}, {}) == tex_src
            c1, c2, c3 = st.columns(3)
            c1.metric("Bullets", len(doc.bullets))
            c2.metric("Sections", len(doc.sections))
            c3.metric("Skill rows", len(doc.skill_lines))
            if ok:
                st.success(
                    f"Parsed `{doc.doc_class or 'unknown class'}` and reproduced your "
                    f"file byte-for-byte. Only bullet text and skill values will change."
                )
            else:
                st.error("Round-trip check failed — this file will not be modified.")
            for w in doc.warnings:
                st.warning(w)
            with st.expander("What the parser found"):
                for s in doc.sections:
                    st.markdown(f"**{s.name}** → `{s.norm}`")
                for e in doc.entries:
                    if e.bullet_ids:
                        st.markdown(f"- {e.label} · {e.dates} · {len(e.bullet_ids)} bullets")

    with right:
        st.subheader("The job posting")
        jd_url = st.text_input("Posting URL (LinkedIn, Naukri, Greenhouse, Lever, …)",
                               placeholder="https://…")
        jd_paste = st.text_area(
            "…or paste the description",
            height=200,
            placeholder="LinkedIn and Naukri block cloud servers, so pasting is "
                        "often the faster path.",
        )
        if st.button("Load posting", type="primary", use_container_width=True):
            with st.spinner("Reading the posting…"):
                job = fetch(jd_paste.strip() or jd_url.strip())
            st.session_state["job"] = job
            st.session_state["jd"] = None

        job: Optional[JobPosting] = st.session_state.get("job")
        if job:
            if job.ok:
                st.success(f"Loaded {job.word_count} words via `{job.source}`.")
                if job.header():
                    st.markdown(f"**{job.header()}**")
                with st.expander("Posting text"):
                    st.text(job.text[:9000])
            else:
                st.error(job.error)

    st.divider()
    st.subheader("Skills you have that the resume under-sells")
    st.caption(
        "This is the honesty valve. Tick anything you have genuinely used — the "
        "agent may then write it in. Anything not ticked and not already on your "
        "resume is treated as off-limits, and a rewrite that claims it gets "
        "rejected automatically."
    )
    doc: Optional[ResumeDoc] = st.session_state.get("doc")
    job = st.session_state.get("job")
    suggestions: List[str] = []
    if doc and job and job.ok:
        jd_preview = analyse(job.text, None, title_hint=job.title, company_hint=job.company)
        on_resume = " , ".join(doc.all_skill_values()) + " " + doc.plain_text()
        suggestions = [
            S.display_name(r.skill) for r in jd_preview.must_haves() + jd_preview.nice_to_haves()
            if not S.mentioned_in(r.skill, on_resume)
        ][:24]

    declared = st.multiselect(
        "I have used these, even though they are not on my resume",
        options=sorted(set(suggestions)) or sorted({S.display_name(k) for k in S.ALIASES}),
        default=[],
    )
    extra_declared = st.text_input(
        "Anything else (comma separated)", placeholder="Kafka Streams, Resilience4j")
    context_notes = st.text_area(
        "True details about your work that are not written on the resume yet",
        height=110,
        placeholder="e.g. The Home Loan backend peaked at 40K applications a month.\n"
                    "The EFE services run Java 21 and Spring Boot 4.\n"
                    "I mentored two juniors through their first on-call rotation.",
        help="Numbers you put here become usable. Numbers you do not put here, and "
             "that are not already on your resume, can never appear in the output.",
    )
    st.session_state["declared"] = declared + [
        s.strip() for s in extra_declared.split(",") if s.strip()]
    st.session_state["context_notes"] = context_notes


# --------------------------------------------------------------------------
# 2. gap analysis
# --------------------------------------------------------------------------

def _score_card(sc: ATSScore, label: str, delta: Optional[int] = None) -> None:
    st.markdown(f"**{label}**")
    st.markdown(
        f"<div class='metricbig'>{sc.total}"
        + (f" <span style='font-size:1rem;color:{'#2f9e4f' if delta and delta > 0 else '#888'}'>"
           f"{'+' if delta and delta > 0 else ''}{delta}</span>" if delta is not None else "")
        + f"</div><div class='muted'>{sc.band()}</div>",
        unsafe_allow_html=True,
    )
    for k, v in sc.breakdown.items():
        st.progress(min(1.0, v / WEIGHTS[k]), text=f"{k.replace('_',' ').title()} {v:.0f}/{WEIGHTS[k]}")


with tab_gaps:
    doc = st.session_state.get("doc")
    job = st.session_state.get("job")
    if not doc or not job or not job.ok:
        st.info("Load a resume and a job posting on the Inputs tab first.")
    else:
        cl = client_for(model)
        # The gap report itself never needs a model. The only optional model
        # call here refines the JD's requirement list — worth a second against
        # a hosted endpoint, but a local 8B takes minutes, which turns the tab
        # that is supposed to be instant into the slowest one in the app.
        use_llm_jd = False
        if cl.available:
            use_llm_jd = st.checkbox(
                "Use the model to refine the job requirements (slower)",
                value=not cl.provider.local,
                help=("The gap report, evidence grading and ATS score are all "
                      "computed offline and need no model. This option only "
                      "improves how the posting's requirements are extracted. "
                      "On a local model it costs minutes for a modest gain — "
                      "the tailoring step re-reads the posting anyway."),
            )

        if st.button("Analyse the gap", type="primary"):
            spin = ("Reading the posting with the model…" if use_llm_jd
                    else "Reading the posting for what it actually demands…")
            with st.spinner(spin):
                jd = analyse(job.text, cl if use_llm_jd else None,
                             title_hint=job.title, company_hint=job.company)
            report = build_report(doc, jd, st.session_state.get("declared", []))
            st.session_state["jd"] = jd
            st.session_state["report"] = report
            st.session_state["ats_before"] = ats_score(doc, jd, report)
            st.session_state["result"] = None

        jd: Optional[JDAnalysis] = st.session_state.get("jd")
        report: Optional[GapReport] = st.session_state.get("report")
        if jd and report:
            if not jd.used_llm:
                if jd.llm_error:
                    # A key was present and the call still failed. Telling
                    # people to "add a key" here sent them hunting for one they
                    # already had — the real reason was a retired model name.
                    st.error(
                        f"**The model pass failed, so this is the offline "
                        f"vocabulary analysis.**\n\n{jd.llm_error}\n\n"
                        f"The gap report below is still real — it just finds "
                        f"the obvious keywords and misses nuance. Fix the "
                        f"above and press *Analyse the gap* again."
                    )
                elif not use_llm_jd:
                    st.info(
                        "Offline vocabulary analysis — the model pass is "
                        "switched off above. Tick the box and re-run for the "
                        "full reading."
                    )
                else:
                    st.warning(
                        "Running without an API key, so this is the vocabulary-based "
                        "fallback analysis. It finds the obvious keywords and misses "
                        "the nuance. Add a key for the real thing."
                    )
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                _score_card(st.session_state["ats_before"], "ATS score, as it stands")
                cov = report.coverage()
                st.caption(
                    f"Must-have coverage {cov['must_have']:.0%} · "
                    f"Nice-to-have {cov['nice_to_have']:.0%}"
                )
                if jd.red_flags:
                    with st.expander("Things to know about this posting"):
                        for f in jd.red_flags:
                            st.markdown(f"- {f}")
            with c2:
                st.markdown("#### What the posting wants, and what you can prove")
                for g in report.gaps[:26]:
                    icon = LEVEL_ICON.get(g.level, "·")
                    color = LEVEL_COLOR.get(g.level, "#555")
                    tag = "MUST" if g.is_must else "nice"
                    where = (" · in " + ", ".join(g.bullet_ids)) if g.bullet_ids else ""
                    st.markdown(
                        f"<div class='gaprow' style='border-left-color:{color}'>"
                        f"{icon} <b>{S.display_name(g.skill)}</b> "
                        f"<span class='pill' style='background:{color}'>{tag}</span>"
                        f"<span class='muted'>{LEVEL_LABEL[g.level]}{where}</span><br>"
                        f"<span class='muted'>{g.action}</span></div>",
                        unsafe_allow_html=True,
                    )

            hard = report.true_gaps()
            if hard:
                st.divider()
                st.markdown("#### Gaps no rewrite can close")
                st.caption(
                    "These are genuinely absent. The agent will not invent them. "
                    "Decide whether to apply anyway or close one first."
                )
                for g in hard[:10]:
                    st.markdown(f"- 🔴 **{S.display_name(g.skill)}** — {g.action}")

            checks = st.session_state["ats_before"].checks
            failed = [c for c in checks if not c.passed]
            if failed:
                st.divider()
                st.markdown("#### Format issues")
                for c in failed:
                    st.markdown(f"- **{c.label}** — {c.detail}"
                                + (f" `{', '.join(c.items[:6])}`" if c.items else ""))


# --------------------------------------------------------------------------
# 3. tailor
# --------------------------------------------------------------------------

with tab_tailor:
    doc = st.session_state.get("doc")
    jd = st.session_state.get("jd")
    report = st.session_state.get("report")
    if not (doc and jd and report):
        st.info("Run the gap analysis first.")
    else:
        cl = client_for(model)
        if not cl.available:
            st.error(
                f"Tailoring needs a {cl.provider.label} API key — add one in the "
                f"sidebar, or switch to Ollama (local, free, no key) there. "
                f"Everything on the Gap analysis tab works without any key."
            )
        else:
            if st.button("Tailor my resume", type="primary"):
                bar = st.progress(0.0, text="Starting…")

                def on_progress(msg: str, frac: float) -> None:
                    bar.progress(min(1.0, frac), text=msg)

                res = tailor(
                    doc, jd, cl,
                    declared_skills=st.session_state.get("declared", []),
                    context_notes=st.session_state.get("context_notes", ""),
                    max_rounds=rounds, progress=on_progress,
                )
                bar.empty()
                st.session_state["result"] = res
                st.session_state["accepted"] = {
                    c.bid: True for c in (res.best.changes if res.best else []) if c.changed
                }
                st.session_state["pdf_after"] = None
                for k in ("cover", "outreach", "prep", "plan"):
                    st.session_state[k] = None

            res: Optional[TailorResult] = st.session_state.get("result")
            if res and res.error:
                st.error(res.error)
            elif res and res.best:
                best = res.best
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ATS score", best.ats.total,
                          best.ats.total - res.ats_before.total)
                c2.metric("Must-have coverage",
                          f"{best.report_after.coverage()['must_have']:.0%}",
                          f"{(best.report_after.coverage()['must_have'] - res.report_before.coverage()['must_have']):+.0%}")
                c3.metric("Bullets changed", best.changed_count())
                c4.metric("Critique rounds", len(res.drafts))

                if best.critique and best.critique.verdict:
                    st.info(f"**Critic's verdict** — {best.critique.verdict}")

                if best.validation and best.validation.violations:
                    with st.expander(
                            f"⚠️ {len(best.validation.violations)} rewrite(s) were blocked "
                            f"and reverted to your original", expanded=True):
                        st.caption(
                            "These failed the truth checks, so your original text was "
                            "kept. This is the system working, not an error."
                        )
                        for v in best.validation.violations:
                            st.markdown(f"- `{v.bullet_id}` **{v.kind}** — {v.message}")

                if best.ats.stuffing_warning:
                    st.warning(best.ats.stuffing_warning)

                if best.critique and best.critique.fabrication_flags:
                    st.error("The critic flagged unsupported claims — review these closely:")
                    for f in best.critique.fabrication_flags:
                        st.markdown(f"- `{f.get('id','')}` “{f.get('claim','')}” — {f.get('why','')}")

                st.divider()
                st.markdown("#### Every change, with the reasoning")
                st.caption("Untick anything you do not want. The download follows your choices.")

                changes = [c for c in best.changes if c.changed]
                if not changes:
                    st.info("The agent decided your resume already fits this posting.")
                for ch in changes:
                    entry = doc.entry(doc.bullet(ch.bid).entry_key) if doc.bullet(ch.bid) else None
                    added, removed = change_stats(ch.original, ch.tailored)
                    with st.container():
                        st.markdown("<div class='bulletbox'>", unsafe_allow_html=True)
                        head = st.columns([6, 1])
                        head[0].markdown(
                            f"**{entry.label if entry else ch.bid}** "
                            f"<span class='muted'>· {ch.bid} · +{added}/−{removed} words</span>",
                            unsafe_allow_html=True)
                        st.session_state["accepted"][ch.bid] = head[1].checkbox(
                            "Use", value=st.session_state["accepted"].get(ch.bid, True),
                            key=f"acc_{ch.bid}")
                        st.markdown(to_html(ch.original, ch.tailored), unsafe_allow_html=True)
                        if ch.rationale:
                            st.caption(f"Why: {ch.rationale}")
                        if ch.truth_basis:
                            st.caption(f"Grounded in: {ch.truth_basis}")
                        if ch.keywords:
                            st.caption("Keywords placed: " + ", ".join(ch.keywords))
                        st.markdown("</div>", unsafe_allow_html=True)

                if best.skill_edits:
                    st.markdown("#### Skills section")
                    for sid, new_vals in best.skill_edits.items():
                        line = doc.skill_line(sid)
                        if line:
                            st.markdown(f"**{line.label}**")
                            st.markdown(to_html(line.raw.strip(), new_vals.strip()),
                                        unsafe_allow_html=True)

                with st.expander("Agent log"):
                    for line in res.log:
                        st.text("· " + line)
                    st.caption(
                        f"{res.usage.calls} model calls · "
                        f"{res.usage.input_tokens:,} in / {res.usage.output_tokens:,} out"
                        + (f" · about ${res.usage.estimated_cost_usd(cl.provider.in_rate, cl.provider.out_rate):.3f}"
                           if cl.provider.in_rate
                           else (" · local, no cost" if cl.provider.local
                                 else " · free tier"))
                    )


# --------------------------------------------------------------------------
# 4. download
# --------------------------------------------------------------------------

def _final_tex(doc: ResumeDoc, best: Draft, accepted: Dict[str, bool]) -> str:
    bullets = {k: v for k, v in best.bullet_edits.items() if accepted.get(k, True)}
    return render(doc, bullets, best.skill_edits)


with tab_output:
    doc = st.session_state.get("doc")
    res = st.session_state.get("result")
    if not (doc and res and res.best):
        st.info("Tailor the resume first.")
    else:
        best = res.best
        final_tex = _final_tex(doc, best, st.session_state.get("accepted", {}))
        company_slug = re.sub(r"[^a-z0-9]+", "-",
                              (res.jd.company or "role").lower()).strip("-") or "role"
        stem = f"resume-{company_slug}-{datetime.now():%Y%m%d}"

        st.success(
            "Your preamble, document class, macros and section scaffolding are "
            "byte-for-byte identical to what you uploaded."
        )

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Tailored .tex", final_tex, f"{stem}.tex",
                           "text/x-tex", type="primary", use_container_width=True)
        c2.download_button("⬇️ Diff vs original",
                           unified(doc.source, final_tex, st.session_state["tex_name"] or "resume.tex"),
                           f"{stem}.diff", "text/plain", use_container_width=True)

        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{stem}.tex", final_tex)
            z.writestr("original.tex", doc.source)
            z.writestr("gap-report.json", json.dumps({
                "job": {"title": res.jd.role_title, "company": res.jd.company},
                "ats_before": res.ats_before.total,
                "ats_after": best.ats.total,
                "coverage_before": res.report_before.coverage(),
                "coverage_after": best.report_after.coverage(),
                "true_gaps": [S.display_name(g.skill) for g in res.report_before.true_gaps()],
                "changes": [{"id": c.bid, "before": c.original, "after": c.tailored,
                             "why": c.rationale} for c in best.changes if c.changed],
            }, indent=2))
            for name, blob in st.session_state.get("cls_files", {}).items():
                z.writestr(name, blob)
        c3.download_button("⬇️ Full bundle (.zip)", bundle.getvalue(),
                           f"{stem}.zip", "application/zip", use_container_width=True)

        st.divider()
        st.markdown("#### PDF preview")
        if not available_engines():
            st.info(
                "No LaTeX engine on this server. Download the .tex and compile it "
                "in Overleaf — it uses your template unchanged, so it will build "
                "exactly as your current resume does."
            )
        else:
            if st.button("Build PDF"):
                support = dict(st.session_state.get("cls_files", {}))
                used_shim = False
                if doc.doc_class and f"{doc.doc_class}.cls" not in support:
                    support[f"{doc.doc_class}.cls"] = shim_class(doc.doc_class).encode()
                    used_shim = True
                with st.spinner("Compiling…"):
                    out = compile_tex(final_tex, support, engine_hint=doc.engine_hint)
                st.session_state["pdf_after"] = out
                st.session_state["pdf_shim"] = used_shim

            out = st.session_state.get("pdf_after")
            if out:
                if out.ok:
                    if st.session_state.get("pdf_shim"):
                        st.warning(
                            "Built with a stand-in class file, so fonts and spacing "
                            "will not match your real resume. Upload your .cls in the "
                            "sidebar for an exact preview."
                        )
                    st.download_button("⬇️ PDF", out.pdf, f"{stem}.pdf",
                                       "application/pdf", use_container_width=True)
                    st.markdown("##### ATS parseability, tested on the actual PDF")
                    for c in parseability_from_pdf(out.pdf, parse_tex(final_tex)):
                        icon = "✅" if c.passed else ("🔴" if c.severity == "fail" else "🟠")
                        st.markdown(f"{icon} **{c.label}** — {c.detail}")
                else:
                    st.error(out.friendly_error())
                    with st.expander("LaTeX log"):
                        st.code(out.log[-4000:])

        with st.expander("Final .tex"):
            st.code(final_tex, language="latex")


# --------------------------------------------------------------------------
# 5. after you apply
# --------------------------------------------------------------------------

with tab_next:
    doc = st.session_state.get("doc")
    res = st.session_state.get("result")
    jd = st.session_state.get("jd")
    report = st.session_state.get("report")
    if not (doc and res and res.best and jd):
        st.info("Tailor the resume first.")
    else:
        cl = client_for(model)
        name = ""
        m = re.search(r"\\name\s*\{([^{}]+)\}", doc.source)
        if m:
            name = m.group(1).strip()

        sub1, sub2, sub3, sub4 = st.tabs(
            ["Cover note", "Recruiter message", "Interview risk", "Gap plan"])

        with sub1:
            if st.button("Write a cover note"):
                with st.spinner("Writing…"):
                    st.session_state["cover"] = extras.cover_letter(
                        doc, jd, report, cl, res.best, name)
            cov = st.session_state.get("cover")
            if cov:
                st.text_input("Subject", cov.subject)
                st.text_area("Body", cov.body, height=290)

        with sub2:
            if st.button("Draft outreach messages"):
                with st.spinner("Writing…"):
                    st.session_state["outreach"] = extras.outreach(
                        doc, jd, report, cl, res.best, name,
                        current_role=doc.entries[0].title if doc.entries else "")
            o = st.session_state.get("outreach")
            if o:
                st.text_area("To a recruiter", o.to_recruiter, height=140)
                st.text_area("To the hiring manager", o.to_hiring_manager, height=140)
                st.text_area("Connection request note", o.connection_note, height=90)

        with sub3:
            st.caption(
                "Every bullet you sharpened is a question you have invited. "
                "This is what they will ask."
            )
            if st.button("Show interview risk"):
                with st.spinner("Thinking like an interviewer…"):
                    st.session_state["prep"] = extras.interview_prep(res.best, jd, cl)
            prep = st.session_state.get("prep")
            if prep:
                if prep.likely_deep_dive:
                    st.info(f"**Where they will push hardest** — {prep.likely_deep_dive}")
                for b in prep.per_bullet:
                    risk = str(b.get("risk", "")).lower()
                    icon = {"high": "🔴", "medium": "🟠"}.get(risk, "🟢")
                    with st.expander(f"{icon} {b.get('id','')} — {b.get('claim','')[:80]}"):
                        for q in b.get("questions", []):
                            st.markdown(f"- {q}")
                        if b.get("prep_note"):
                            st.caption(f"Be ready to say: {b['prep_note']}")
                if prep.study_list:
                    st.markdown("**Revise before the screen**")
                    for s in prep.study_list:
                        st.markdown(f"- {s}")

        with sub4:
            st.caption("For the gaps no rewrite could close.")
            if st.button("Build a plan"):
                with st.spinner("Planning…"):
                    st.session_state["plan"] = extras.learning_plan(report, jd, cl)
            plan = st.session_state.get("plan")
            if plan:
                if plan.get("verdict"):
                    st.info(plan["verdict"])
                for p in plan.get("plans", []):
                    worth = "worth doing" if p.get("worth_it") else "skip it"
                    with st.expander(f"{p.get('skill','')} — {worth}, {p.get('effort','')}"):
                        st.markdown(f"**Project** — {p.get('project','')}")
                        if p.get("resulting_bullet"):
                            st.markdown(f"**You could then write** — *{p['resulting_bullet']}*")
                        if p.get("interview_bridge"):
                            st.caption(f"If asked before then: {p['interview_bridge']}")
