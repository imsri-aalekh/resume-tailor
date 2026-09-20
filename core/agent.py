"""
The tailoring agent: a writer, an adversarial critic, and a reviser in a loop.

Why a loop and not one big prompt: a single pass optimises for the job
description and quietly loses the things that made the resume good — the
numbers, the named systems, the specificity. A separate critic that has never
seen the writer's reasoning catches that, because its only job is to attack the
draft. We run writer → validate → critic → revise → validate → critic, keep the
best-scoring draft, and report exactly what changed and why.

Every draft passes through `guards.validate_rewrites` before it is scored.
Anything that fails reverts to the original bullet and is handed to the critic
as a violation it must resolve, so the loop cannot converge on a lie.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from . import skills as S
from .ats import ATSScore, score as ats_score
from .guards import (ValidationResult, _numbers, allowed_numbers, skills_in,
                     validate_rewrites, validate_skill_lines)
from .jd_extract import JDAnalysis
from .latexdoc import ResumeDoc, parse as parse_tex, render, verify_template_integrity
from .llm import LLMClient, LLMError, Usage
from .matcher import GapReport, build_report

# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

WRITER_SYSTEM = """You tailor an experienced software engineer's resume to a \
specific job description. You are a careful editor, not a copywriter.

Absolute rules, in priority order:

1. NEVER claim anything the candidate has not done. You may re-frame, re-order \
and re-emphasise existing work. You may not invent projects, tools, scale, \
teams or outcomes.
2. NEVER invent a number. Every figure in your output must already appear in \
the candidate's resume. If a bullet has no metric, leave it without one rather \
than guessing.
3. Only mention technologies from the ALLOWED VOCABULARY. That list is the \
union of what is already on the resume and what the candidate personally \
confirmed they have used.
4. Keep every named entity — companies, products, vendors, internal system \
names. They are what make a bullet credible.
5. Mirror the job posting's exact spelling of a keyword ("Apache Kafka", \
"CI/CD", "REST APIs") so keyword matching lands.
6. One bullet = one accomplishment. Lead with a strong past-tense verb. \
Under 40 words. No first person.
7. Output LaTeX-safe text: escape %, &, #, _ as \\%, \\&, \\#, \\_. Do not use \
any macro other than \\textbf{} and keep even that rare.

If a bullet is already well-targeted, return it unchanged and say so. Changing \
everything is a failure mode, not thoroughness."""

WRITER_PROMPT = """Tailor these resume bullets for the role below.

## TARGET ROLE
{role_line}
Seniority: {seniority}   Domain: {domain}

## WHAT THE POSTING DEMANDS (highest weight first)
{requirements}

## KEYWORDS TO MIRROR VERBATIM WHERE TRUE
{ats_keywords}

## GAPS THE REWRITE SHOULD CLOSE
These are skills the candidate genuinely has but the resume under-sells. Work \
each one into the bullet where it actually happened:
{gap_lines}

## ALLOWED VOCABULARY (nothing outside this list may appear)
{vocabulary}

## NUMBERS THAT EXIST IN THE RESUME (no others may be used)
{numbers}

## ADDITIONAL TRUE CONTEXT FROM THE CANDIDATE
Facts the candidate confirmed about this work that are not yet written on the
resume. You may use these, and only these, beyond what the bullets already say:
{context_notes}

## THE BULLETS
{bullets}

{feedback_block}

Return JSON:
{{
  "edits": [
    {{
      "id": "b001",
      "new_text": "the rewritten bullet, LaTeX-safe",
      "changed": true,
      "keywords_placed": ["Apache Kafka"],
      "rationale": "one short line: what you changed and which requirement it serves",
      "truth_basis": "which part of the original bullet supports this claim"
    }}
  ],
  "skills_line_edits": [
    {{"id": "s001", "new_values": "comma separated values", "added": ["..."]}}
  ],
  "untouched_reason": "why you left the remaining bullets alone"
}}

Include an entry for every bullet you changed. Omit bullets you left alone."""


CRITIC_SYSTEM = """You are two people at once.

First, a hiring manager at the target company who has 90 seconds and 200 \
resumes. You are looking for a reason to say no.

Second, the interviewer who will have to take this candidate through a deep \
dive on anything the resume claims. You are asking: if I probe this bullet for \
ten minutes, does it hold up, or does it collapse?

Judge the tailored draft against the original. You are looking specifically for:

- **Fabrication or inflation.** Anything claimed that the original does not \
support. This is disqualifying. Flag it even if it is subtle — "led" where the \
original said "worked with", "architected" where it said "implemented", scale \
that appeared from nowhere.
- **Keyword stuffing.** Technology names dropped into bullets where they do not \
belong, or a bullet that reads as a tech list instead of an accomplishment.
- **Lost substance.** A rewrite that gained a keyword and lost the metric, the \
named system, or the actual outcome. This is the most common failure.
- **Vagueness.** Bullets that could describe any engineer at any company.
- **Missed opportunity.** A requirement the candidate demonstrably meets that \
the draft still does not surface.

Be specific and be hard to please. A draft with no criticism is a draft you did \
not read carefully. Score honestly — most first drafts deserve 60-75."""

CRITIC_PROMPT = """## TARGET ROLE
{role_line}

## WHAT THE POSTING DEMANDS
{requirements}

## ALLOWED VOCABULARY (anything outside this list is fabrication)
{vocabulary}

## VALIDATOR VIOLATIONS ALREADY DETECTED
These rewrites were rejected automatically and reverted to the original. Your \
fixes must resolve the underlying problem, not repeat it:
{violations}

## THE DRAFT (original → tailored)
{comparison}

## CURRENT MEASURED COVERAGE
{coverage}

Return JSON:
{{
  "overall_score": 0-100,
  "verdict": "one paragraph: would this get an interview, and what is the single biggest problem",
  "bullet_verdicts": [
    {{
      "id": "b001",
      "verdict": "keep | revise | revert",
      "issue": "what is wrong, specific",
      "fix": "the concrete change to make",
      "severity": "critical | major | minor"
    }}
  ],
  "fabrication_flags": [
    {{"id": "b001", "claim": "the exact phrase", "why": "what the original actually supports"}}
  ],
  "missed_opportunities": [
    {{"requirement": "Kafka", "where": "b007", "suggestion": "what to say"}}
  ],
  "global_notes": ["resume-level observations"]
}}"""


REVISER_PROMPT = """Revise the draft using the critic's feedback.

Apply every critical and major fix. Where the critic says "revert", restore the \
original bullet exactly. Where it flags fabrication, remove the unsupported \
claim entirely — do not soften it.

The same absolute rules apply: no invented numbers, nothing outside the allowed \
vocabulary, keep named entities, under 40 words, LaTeX-safe.

## CRITIC FEEDBACK
{critique}

## VALIDATOR VIOLATIONS (must be resolved)
{violations}

## ALLOWED VOCABULARY
{vocabulary}

## NUMBERS AVAILABLE
{numbers}

## ADDITIONAL TRUE CONTEXT FROM THE CANDIDATE
{context_notes}

## CURRENT STATE (original → current draft)
{comparison}

Return the same JSON shape as before:
{{"edits": [{{"id": "", "new_text": "", "changed": true, "keywords_placed": [], "rationale": "", "truth_basis": ""}}],
  "skills_line_edits": [{{"id": "", "new_values": "", "added": []}}]}}"""


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass
class BulletChange:
    bid: str
    original: str
    tailored: str
    rationale: str = ""
    truth_basis: str = ""
    keywords: List[str] = field(default_factory=list)
    accepted: bool = True

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.tailored.strip()


@dataclass
class Critique:
    overall_score: float = 0.0
    verdict: str = ""
    bullet_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    fabrication_flags: List[Dict[str, Any]] = field(default_factory=list)
    missed_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    global_notes: List[str] = field(default_factory=list)

    def critical_count(self) -> int:
        return sum(1 for v in self.bullet_verdicts
                   if str(v.get("severity", "")).lower() == "critical")


@dataclass
class Draft:
    round_no: int
    bullet_edits: Dict[str, str] = field(default_factory=dict)
    skill_edits: Dict[str, str] = field(default_factory=dict)
    changes: List[BulletChange] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    critique: Optional[Critique] = None
    tex: str = ""
    ats: Optional[ATSScore] = None
    report_after: Optional[GapReport] = None
    integrity: List[str] = field(default_factory=list)
    composite: float = 0.0
    notes: List[str] = field(default_factory=list)

    def changed_count(self) -> int:
        return sum(1 for c in self.changes if c.changed)


@dataclass
class TailorResult:
    best: Optional[Draft] = None
    drafts: List[Draft] = field(default_factory=list)
    report_before: Optional[GapReport] = None
    ats_before: Optional[ATSScore] = None
    jd: Optional[JDAnalysis] = None
    log: List[str] = field(default_factory=list)
    error: str = ""
    # Snapshotted here because Streamlit rebuilds the client on every rerun —
    # reading usage off a live client would show zero the moment the user
    # ticks a checkbox.
    usage: Usage = field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return self.best is not None and not self.error


# --------------------------------------------------------------------------
# prompt assembly helpers
# --------------------------------------------------------------------------


def _fmt_requirements(jd: JDAnalysis, limit: int = 22) -> str:
    rows = []
    for r in (jd.must_haves() + jd.nice_to_haves())[:limit]:
        tag = "MUST" if r.kind == "must" else "nice"
        ev = f"  — \"{r.evidence[:110]}\"" if r.evidence else ""
        rows.append(f"- [{tag} w{r.weight}] {r.skill}{ev}")
    return "\n".join(rows) or "- (none extracted)"


def _fmt_gaps(report: GapReport, limit: int = 12) -> str:
    rows = []
    for g in report.actionable()[:limit]:
        where = f" (currently in: {', '.join(g.bullet_ids)})" if g.bullet_ids else ""
        rows.append(f"- {S.display_name(g.skill)} [{g.level}, priority {g.priority}]"
                    f"{where}: {g.action}")
    return "\n".join(rows) or "- (no actionable gaps — the resume already covers this posting)"


def _fmt_bullets(doc: ResumeDoc, editable: Optional[Set[str]] = None) -> str:
    out: List[str] = []
    current_entry = None
    for b in doc.bullets:
        if b.section not in ("experience", "projects"):
            continue
        if editable is not None and b.bid not in editable:
            continue
        entry = doc.entry(b.entry_key)
        label = entry.label if entry else b.section
        dates = f" ({entry.dates})" if entry and entry.dates else ""
        if label != current_entry:
            out.append(f"\n### {label}{dates}  [{b.section}]")
            current_entry = label
        out.append(f"  {b.bid}: {b.text}")
    return "\n".join(out).strip() or "(no editable bullets)"


def _fmt_skill_lines(doc: ResumeDoc) -> str:
    return "\n".join(f"  {s.sid} [{s.label}]: {', '.join(s.values())}"
                     for s in doc.skill_lines) or "  (none)"


def _fmt_comparison(doc: ResumeDoc, edits: Dict[str, str],
                    skill_edits: Dict[str, str]) -> str:
    out: List[str] = []
    for b in doc.bullets:
        if b.section not in ("experience", "projects"):
            continue
        new = edits.get(b.bid)
        if new and new.strip() != b.text.strip():
            out.append(f"{b.bid} ORIGINAL: {b.text}\n{b.bid} TAILORED: {new}")
        else:
            out.append(f"{b.bid} UNCHANGED: {b.text}")
    for s in doc.skill_lines:
        new = skill_edits.get(s.sid)
        if new and new.strip() != s.raw.strip():
            out.append(f"{s.sid} ORIGINAL SKILLS [{s.label}]: {s.raw.strip()}\n"
                       f"{s.sid} TAILORED SKILLS [{s.label}]: {new.strip()}")
    return "\n\n".join(out)


def _fmt_violations(val: Optional[ValidationResult]) -> str:
    if val is None or not val.violations:
        return "(none)"
    return "\n".join(f"- {v.bullet_id} [{v.kind}] {v.message}" for v in val.violations[:30])


def _fmt_vocab(vocab: Set[str]) -> str:
    return ", ".join(sorted(S.display_name(v) for v in vocab))


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------


def _parse_edits(data: Any) -> tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    bullet_edits: Dict[str, str] = {}
    skill_edits: Dict[str, str] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return bullet_edits, skill_edits, meta

    for e in data.get("edits") or []:
        if not isinstance(e, dict):
            continue
        bid = str(e.get("id") or "").strip()
        txt = e.get("new_text")
        if not bid or not txt:
            continue
        if e.get("changed") is False:
            continue
        bullet_edits[bid] = str(txt).strip()
        meta[bid] = {
            "rationale": str(e.get("rationale", "")),
            "truth_basis": str(e.get("truth_basis", "")),
            "keywords": [str(k) for k in (e.get("keywords_placed") or [])],
        }

    for e in data.get("skills_line_edits") or []:
        if not isinstance(e, dict):
            continue
        sid = str(e.get("id") or "").strip()
        vals = e.get("new_values")
        if sid and vals:
            skill_edits[sid] = str(vals).strip()
    return bullet_edits, skill_edits, meta


def _parse_critique(data: Any) -> Critique:
    c = Critique()
    if not isinstance(data, dict):
        return c
    try:
        c.overall_score = float(data.get("overall_score") or 0)
    except (TypeError, ValueError):
        c.overall_score = 0.0
    c.verdict = str(data.get("verdict", ""))
    for key, attr in (("bullet_verdicts", "bullet_verdicts"),
                      ("fabrication_flags", "fabrication_flags"),
                      ("missed_opportunities", "missed_opportunities")):
        vals = data.get(key) or []
        setattr(c, attr, [v for v in vals if isinstance(v, dict)])
    c.global_notes = [str(n) for n in (data.get("global_notes") or [])]
    return c


# --------------------------------------------------------------------------
# scoring a draft
# --------------------------------------------------------------------------


def _build_draft(doc: ResumeDoc, jd: JDAnalysis, report: GapReport,
                 round_no: int, bullet_edits: Dict[str, str],
                 skill_edits: Dict[str, str],
                 meta: Dict[str, Dict[str, Any]],
                 vocab: Set[str],
                 extra_numbers: Optional[Set[str]] = None) -> Draft:
    val_b = validate_rewrites(doc, bullet_edits, vocab, extra_numbers)
    val_s = validate_skill_lines(doc, skill_edits, vocab)

    combined = ValidationResult(
        accepted={**val_b.accepted, **val_s.accepted},
        rejected={**val_b.rejected, **val_s.rejected},
        violations=val_b.violations + val_s.violations,
    )

    safe_bullets = dict(val_b.accepted)
    safe_skills = dict(val_s.accepted)

    tex = render(doc, safe_bullets, safe_skills)
    integrity = verify_template_integrity(doc, tex, safe_bullets, safe_skills)

    new_doc = parse_tex(tex)
    report_after = build_report(new_doc, jd, report.declared_skills)
    ats_after = ats_score(new_doc, jd, report_after)

    changes: List[BulletChange] = []
    for b in doc.bullets:
        proposed = bullet_edits.get(b.bid)
        final = safe_bullets.get(b.bid, b.text)
        if proposed is None and final == b.text:
            continue
        m = meta.get(b.bid, {})
        changes.append(BulletChange(
            bid=b.bid, original=b.text, tailored=final,
            rationale=m.get("rationale", ""),
            truth_basis=m.get("truth_basis", ""),
            keywords=m.get("keywords", []),
            accepted=b.bid in safe_bullets,
        ))

    return Draft(round_no=round_no, bullet_edits=safe_bullets, skill_edits=safe_skills,
                 changes=changes, validation=combined, tex=tex, ats=ats_after,
                 report_after=report_after, integrity=integrity)


def _composite(draft: Draft, ats_before: ATSScore) -> float:
    """Objective-first selection metric.

    Measured ATS improvement dominates; the critic's opinion is a tiebreaker.
    Violations and fabrication flags are punished hard enough to never win.
    """
    ats_now = draft.ats.total if draft.ats else 0
    gain = ats_now - ats_before.total
    critic = draft.critique.overall_score if draft.critique else 55.0

    score = 0.55 * ats_now + 0.25 * critic + 1.2 * max(0, gain)

    n_viol = len(draft.validation.violations) if draft.validation else 0
    score -= 6.0 * n_viol
    if draft.critique:
        score -= 9.0 * len(draft.critique.fabrication_flags)
        score -= 4.0 * draft.critique.critical_count()
    if draft.integrity:
        score -= 40.0
    if draft.changed_count() == 0:
        score -= 12.0
    if draft.ats and draft.ats.stuffing_warning:
        score -= 8.0
    return round(score, 2)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def tailor(doc: ResumeDoc, jd: JDAnalysis, client: LLMClient, *,
           declared_skills: Optional[List[str]] = None,
           context_notes: str = "",
           max_rounds: int = 2,
           target_score: int = 82,
           editable_sections: tuple = ("experience", "projects"),
           progress: Optional[Callable[[str, float], None]] = None) -> TailorResult:
    """Run the full generate → critique → revise loop."""

    def tick(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    result = TailorResult(jd=jd)
    report_before = build_report(doc, jd, declared_skills or [])
    ats_before = ats_score(doc, jd, report_before)
    result.report_before = report_before
    result.ats_before = ats_before

    vocab = report_before.allowed_skill_vocabulary()
    # Anything the candidate typed into the extra-context box is treated as true:
    # its technologies join the vocabulary and its figures join the number
    # allowlist, so "the platform handled 50K applications a month" can be used
    # even though it was never on the resume.
    context_notes = (context_notes or "").strip()
    declared_text = " ".join(declared_skills or [])
    if context_notes:
        vocab |= skills_in(context_notes)
    vocab |= skills_in(declared_text)
    extra_numbers = _numbers(context_notes) | _numbers(declared_text)

    editable = {b.bid for b in doc.bullets if b.section in editable_sections}
    if not editable:
        result.error = "No editable bullets found in the Experience or Projects sections."
        return result

    numbers = ", ".join(sorted(n for n in allowed_numbers(doc) if len(n) <= 8)) or "(none)"

    role_line = " — ".join(x for x in (jd.role_title, jd.company) if x) or "(role not identified)"
    common = {
        "role_line": role_line,
        "seniority": jd.seniority,
        "domain": jd.domain or "unspecified",
        "requirements": _fmt_requirements(jd),
        "ats_keywords": ", ".join(jd.ats_keywords[:28]) or "(none)",
        "gap_lines": _fmt_gaps(report_before),
        "vocabulary": _fmt_vocab(vocab),
        "numbers": numbers,
        "context_notes": context_notes or "(none supplied)",
    }

    # ---- round 1: write ------------------------------------------------
    tick("Writing the first tailored draft…", 0.15)
    bullets_block = (_fmt_bullets(doc, editable) +
                     "\n\n### SKILLS ROWS (you may reorder and add confirmed skills)\n" +
                     _fmt_skill_lines(doc))
    try:
        data = client.complete_json(
            WRITER_SYSTEM,
            WRITER_PROMPT.format(bullets=bullets_block, feedback_block="", **common),
            max_tokens=6000, temperature=0.25, label="writer:r1",
        )
    except LLMError as exc:
        result.error = str(exc)
        result.usage = client.usage
        return result

    b_edits, s_edits, meta = _parse_edits(data)
    draft = _build_draft(doc, jd, report_before, 1, b_edits, s_edits, meta, vocab,
                         extra_numbers)
    result.drafts.append(draft)
    result.log.append(f"Round 1: {draft.changed_count()} bullets rewritten, "
                      f"{len(draft.validation.violations)} validator violations.")

    # ---- critique / revise rounds --------------------------------------
    for rnd in range(1, max_rounds + 1):
        tick(f"Critiquing draft {rnd}…", 0.15 + 0.35 * (rnd / max_rounds))
        comparison = _fmt_comparison(doc, draft.bullet_edits, draft.skill_edits)
        cov = json.dumps((draft.report_after or report_before).coverage())
        try:
            cdata = client.complete_json(
                CRITIC_SYSTEM,
                CRITIC_PROMPT.format(
                    role_line=role_line,
                    requirements=common["requirements"],
                    vocabulary=common["vocabulary"],
                    violations=_fmt_violations(draft.validation),
                    comparison=comparison,
                    coverage=cov,
                ),
                max_tokens=4000, temperature=0.15, label=f"critic:r{rnd}",
            )
        except LLMError as exc:
            result.log.append(f"Critic failed in round {rnd}: {exc}")
            break

        draft.critique = _parse_critique(cdata)
        draft.composite = _composite(draft, ats_before)
        result.log.append(
            f"Round {rnd} critique: score {draft.critique.overall_score:.0f}, "
            f"{len(draft.critique.fabrication_flags)} fabrication flag(s), "
            f"composite {draft.composite:.1f}."
        )

        good_enough = (
            draft.critique.overall_score >= target_score
            and not draft.critique.fabrication_flags
            and not draft.validation.violations
            and draft.critique.critical_count() == 0
        )
        if good_enough or rnd == max_rounds:
            break

        tick(f"Revising draft {rnd}…", 0.15 + 0.35 * (rnd / max_rounds) + 0.1)
        try:
            rdata = client.complete_json(
                WRITER_SYSTEM,
                REVISER_PROMPT.format(
                    critique=json.dumps(cdata)[:9000],
                    violations=_fmt_violations(draft.validation),
                    vocabulary=common["vocabulary"],
                    numbers=numbers,
                    context_notes=common["context_notes"],
                    comparison=comparison,
                ),
                max_tokens=6000, temperature=0.2, label=f"reviser:r{rnd}",
            )
        except LLMError as exc:
            result.log.append(f"Reviser failed in round {rnd}: {exc}")
            break

        rb, rs, rmeta = _parse_edits(rdata)
        # carry forward anything the reviser did not re-mention
        merged_b = {**draft.bullet_edits, **rb}
        merged_s = {**draft.skill_edits, **rs}
        merged_meta = {**{k: {"rationale": c.rationale, "truth_basis": c.truth_basis,
                             "keywords": c.keywords}
                          for k, c in ((ch.bid, ch) for ch in draft.changes)},
                       **rmeta}
        draft = _build_draft(doc, jd, report_before, rnd + 1,
                             merged_b, merged_s, merged_meta, vocab, extra_numbers)
        result.drafts.append(draft)
        result.log.append(f"Round {rnd + 1}: {draft.changed_count()} bullets changed, "
                          f"{len(draft.validation.violations)} violations.")

    # ---- pick the winner -----------------------------------------------
    tick("Scoring drafts…", 0.9)
    for d in result.drafts:
        if not d.composite:
            d.composite = _composite(d, ats_before)
    result.usage = client.usage
    result.best = max(result.drafts, key=lambda d: d.composite) if result.drafts else None
    if result.best:
        result.log.append(
            f"Selected draft from round {result.best.round_no} "
            f"(composite {result.best.composite:.1f}, "
            f"ATS {ats_before.total} → {result.best.ats.total})."
        )
    tick("Done.", 1.0)
    return result
