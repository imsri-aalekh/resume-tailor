"""
Everything downstream of the tailored resume.

The resume gets you past the filter. These are the things that convert a filter
pass into a conversation: a cover note that is actually about the company, a
recruiter message short enough to get read, and an honest list of the questions
your newly-sharpened bullets have just invited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import skills as S
from .agent import Draft
from .jd_extract import JDAnalysis
from .latexdoc import ResumeDoc
from .llm import LLMClient, LLMError
from .matcher import GapReport

# --------------------------------------------------------------------------
# cover letter
# --------------------------------------------------------------------------

COVER_SYSTEM = """You write short cover notes for experienced software \
engineers. You write like a person, not like a template.

Rules:
- Under 200 words. Three paragraphs at most.
- Open with why this specific role, not "I am writing to apply".
- One paragraph of concrete evidence drawn only from the resume, with the real \
numbers and the real system names.
- Never claim anything the resume does not support.
- No "I am passionate about", no "fast-paced environment", no "leverage", no \
"synergy", no "I believe I would be a great fit".
- End with a plain sentence, not a flourish."""

COVER_PROMPT = """Write a cover note for this application.

ROLE: {role}
COMPANY: {company}
WHAT THE POSTING EMPHASISES: {emphasis}

THE CANDIDATE'S STRONGEST RELEVANT EVIDENCE (use only this):
{evidence}

CANDIDATE NAME: {name}

Return JSON: {{"subject": "email subject line", "body": "the note"}}"""


# --------------------------------------------------------------------------
# recruiter outreach
# --------------------------------------------------------------------------

OUTREACH_SYSTEM = """You write LinkedIn messages to recruiters and hiring \
managers. Under 90 words, because anything longer is not read. No flattery, no \
"I hope this message finds you well", no buzzwords. Lead with the single most \
relevant thing the candidate has built. Ask one clear question."""

OUTREACH_PROMPT = """Write two LinkedIn messages about this role.

ROLE: {role} at {company}
STRONGEST MATCH: {evidence}
CANDIDATE: {name}, currently {current_role}

Return JSON:
{{"to_recruiter": "...", "to_hiring_manager": "...", "connection_note": "under 300 characters"}}"""


# --------------------------------------------------------------------------
# interview risk
# --------------------------------------------------------------------------

INTERVIEW_SYSTEM = """You are a staff engineer who runs technical screens. You \
are given a resume bullet that was just rewritten to emphasise a technology. \
Your job is to write the questions you would ask to find out whether the \
candidate actually did the thing, or just wrote it down.

Be specific and technical. Generic questions are useless. If a bullet claims a \
circuit breaker, ask about half-open state and what the thresholds were, not \
"tell me about resilience"."""

INTERVIEW_PROMPT = """These bullets were sharpened for a {role} role. For each, \
write the questions an interviewer will now ask, and flag anything that looks \
thin.

{bullets}

THE ROLE'S KEY REQUIREMENTS: {requirements}

Return JSON:
{{
  "per_bullet": [
    {{"id": "b001", "claim": "what it now claims",
      "questions": ["specific technical question"],
      "risk": "low | medium | high",
      "prep_note": "what to be ready to say"}}
  ],
  "likely_deep_dive": "the one area they will push hardest on, and why",
  "study_list": ["concrete things to revise before the screen"]
}}"""


@dataclass
class CoverLetter:
    subject: str = ""
    body: str = ""


@dataclass
class Outreach:
    to_recruiter: str = ""
    to_hiring_manager: str = ""
    connection_note: str = ""


@dataclass
class InterviewPrep:
    per_bullet: List[Dict[str, Any]] = field(default_factory=list)
    likely_deep_dive: str = ""
    study_list: List[str] = field(default_factory=list)

    def high_risk(self) -> List[Dict[str, Any]]:
        return [b for b in self.per_bullet if str(b.get("risk", "")).lower() == "high"]


def _evidence_block(doc: ResumeDoc, report: GapReport, draft: Optional[Draft],
                    limit: int = 8) -> str:
    """The bullets that best answer this posting, tailored text where available."""
    edits = draft.bullet_edits if draft else {}
    scored: List[tuple] = []
    strong = {g.canonical for g in report.gaps
              if g.level in ("strong", "present") and g.weight >= 3}
    for b in doc.bullets:
        if b.section not in ("experience", "projects"):
            continue
        text = edits.get(b.bid, b.text)
        hits = len(S.skills_in(text) & strong) if hasattr(S, "skills_in") else 0
        hits = len({S.canonical(x) for x in S.extract_candidate_skills(text)} & strong)
        entry = doc.entry(b.entry_key)
        scored.append((hits, b.bid, entry.label if entry else "", text))
    scored.sort(key=lambda t: -t[0])
    return "\n".join(f"- [{label}] {text}" for _, _, label, text in scored[:limit])


def cover_letter(doc: ResumeDoc, jd: JDAnalysis, report: GapReport,
                 client: LLMClient, draft: Optional[Draft] = None,
                 candidate_name: str = "") -> CoverLetter:
    emphasis = ", ".join(S.display_name(r.skill) for r in jd.must_haves()[:8])
    try:
        data = client.complete_json(
            COVER_SYSTEM,
            COVER_PROMPT.format(
                role=jd.role_title or "the role",
                company=jd.company or "the company",
                emphasis=emphasis or "(not specified)",
                evidence=_evidence_block(doc, report, draft),
                name=candidate_name or "the candidate",
            ),
            max_tokens=1200, temperature=0.4, label="cover_letter",
        )
    except LLMError as exc:
        return CoverLetter(subject="", body=f"(Could not generate: {exc})")
    if not isinstance(data, dict):
        return CoverLetter()
    return CoverLetter(subject=str(data.get("subject", "")),
                       body=str(data.get("body", "")))


def outreach(doc: ResumeDoc, jd: JDAnalysis, report: GapReport,
             client: LLMClient, draft: Optional[Draft] = None,
             candidate_name: str = "", current_role: str = "") -> Outreach:
    try:
        data = client.complete_json(
            OUTREACH_SYSTEM,
            OUTREACH_PROMPT.format(
                role=jd.role_title or "the role",
                company=jd.company or "the company",
                evidence=_evidence_block(doc, report, draft, limit=4),
                name=candidate_name or "the candidate",
                current_role=current_role or "a senior software engineer",
            ),
            max_tokens=900, temperature=0.4, label="outreach",
        )
    except LLMError:
        return Outreach()
    if not isinstance(data, dict):
        return Outreach()
    return Outreach(
        to_recruiter=str(data.get("to_recruiter", "")),
        to_hiring_manager=str(data.get("to_hiring_manager", "")),
        connection_note=str(data.get("connection_note", "")),
    )


def interview_prep(draft: Draft, jd: JDAnalysis, client: LLMClient) -> InterviewPrep:
    changed = [c for c in draft.changes if c.changed]
    if not changed:
        return InterviewPrep(likely_deep_dive="Nothing changed, so nothing new to defend.")
    bullets = "\n\n".join(
        f"{c.bid}\n  BEFORE: {c.original}\n  AFTER:  {c.tailored}" for c in changed[:12]
    )
    try:
        data = client.complete_json(
            INTERVIEW_SYSTEM,
            INTERVIEW_PROMPT.format(
                role=jd.role_title or "senior backend",
                bullets=bullets,
                requirements=", ".join(S.display_name(r.skill) for r in jd.must_haves()[:12]),
            ),
            max_tokens=3000, temperature=0.3, label="interview_prep",
        )
    except LLMError:
        return InterviewPrep()
    if not isinstance(data, dict):
        return InterviewPrep()
    return InterviewPrep(
        per_bullet=[b for b in (data.get("per_bullet") or []) if isinstance(b, dict)],
        likely_deep_dive=str(data.get("likely_deep_dive", "")),
        study_list=[str(s) for s in (data.get("study_list") or [])],
    )


# --------------------------------------------------------------------------
# learning plan for the real gaps
# --------------------------------------------------------------------------

PLAN_SYSTEM = """You advise experienced engineers on closing a real skills gap \
before an application. You are practical and you respect their time.

For each gap: say whether it is worth closing for this role, what the smallest \
credible piece of work would be, and roughly how long. A weekend project that \
produces one honest resume bullet beats a twelve-week course. If a gap is not \
worth closing, say so."""

PLAN_PROMPT = """The candidate is applying for {role} at {company} and is \
missing these, in priority order:

{gaps}

They already know: {owned}

Return JSON:
{{
  "verdict": "should they apply now, or close a gap first",
  "plans": [
    {{"skill": "", "worth_it": true, "effort": "e.g. one weekend",
      "project": "the smallest thing that produces an honest bullet",
      "resulting_bullet": "what they could truthfully write afterwards",
      "interview_bridge": "how to answer if asked before they have done it"}}
  ]
}}"""


def learning_plan(report: GapReport, jd: JDAnalysis, client: LLMClient,
                  limit: int = 6) -> Dict[str, Any]:
    gaps = report.true_gaps()[:limit]
    if not gaps:
        return {"verdict": "No hard gaps — apply.", "plans": []}
    gap_lines = "\n".join(
        f"- {S.display_name(g.skill)} (priority {g.priority}, "
        f"{'required' if g.is_must else 'preferred'})"
        + (f"; closest existing skill: {', '.join(g.adjacent[:3])}" if g.adjacent else "")
        for g in gaps
    )
    try:
        data = client.complete_json(
            PLAN_SYSTEM,
            PLAN_PROMPT.format(
                role=jd.role_title or "the role",
                company=jd.company or "the company",
                gaps=gap_lines,
                owned=", ".join(sorted(report.resume_skills)[:40]),
            ),
            max_tokens=2500, temperature=0.3, label="learning_plan",
        )
    except LLMError:
        return {"verdict": "", "plans": []}
    return data if isinstance(data, dict) else {"verdict": "", "plans": []}
