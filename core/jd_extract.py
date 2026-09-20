"""
Turn a raw job description into structured requirements.

Uses the LLM when a key is present, and falls back to a vocabulary scan so the
app still gives useful output with no key at all.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import skills as S
from .llm import LLMClient, LLMError

SYSTEM = """You analyse job descriptions for an experienced software engineer \
who is tailoring their resume. You are precise and you never invent \
requirements that are not in the posting.

Rules:
- Extract only what the posting actually says. Do not add "standard" \
requirements you think the role probably wants.
- `must_have` = explicitly required, or listed under requirements/qualifications.
- `nice_to_have` = listed as preferred, bonus, plus, or desirable.
- `ats_keywords` = the exact noun phrases an ATS would scan for, in the \
posting's own spelling (e.g. "Apache Kafka" not "kafka"; "CI/CD" not "ci cd").
- `weight` is 1-5: 5 = named repeatedly or in the job title, 1 = mentioned once \
in passing.
- Keep `evidence` to a short verbatim quote from the posting."""

PROMPT = """Analyse this job posting.

<posting>
{jd}
</posting>

Return JSON with exactly this shape:
{{
  "role_title": "",
  "company": "",
  "seniority": "one of: intern, junior, mid, senior, staff, principal, manager, unclear",
  "years_experience": "e.g. '5+' or '' if unstated",
  "must_have": [{{"skill": "", "evidence": "", "weight": 5}}],
  "nice_to_have": [{{"skill": "", "evidence": "", "weight": 2}}],
  "responsibilities": ["short phrases describing what the person will do"],
  "domain": "e.g. fintech, e-commerce, healthcare, infrastructure, or ''",
  "ats_keywords": ["exact phrases worth mirroring verbatim"],
  "soft_signals": ["e.g. 'mentoring juniors', 'cross-functional collaboration'"],
  "red_flags": ["anything the candidate should know: unclear scope, heavy on-call, etc."]
}}"""


@dataclass
class Requirement:
    skill: str
    evidence: str = ""
    weight: int = 3
    kind: str = "must"          # must | nice

    @property
    def canonical(self) -> str:
        return S.canonical(self.skill)


@dataclass
class JDAnalysis:
    role_title: str = ""
    company: str = ""
    seniority: str = "unclear"
    years_experience: str = ""
    requirements: List[Requirement] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    domain: str = ""
    ats_keywords: List[str] = field(default_factory=list)
    soft_signals: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    used_llm: bool = False
    raw_text: str = ""

    def must_haves(self) -> List[Requirement]:
        return sorted([r for r in self.requirements if r.kind == "must"],
                      key=lambda r: -r.weight)

    def nice_to_haves(self) -> List[Requirement]:
        return sorted([r for r in self.requirements if r.kind == "nice"],
                      key=lambda r: -r.weight)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["requirements"] = [asdict(r) for r in self.requirements]
        return d


def _dedupe(reqs: List[Requirement]) -> List[Requirement]:
    """Collapse aliases: 'Kafka' and 'Apache Kafka' become one requirement."""
    best: Dict[str, Requirement] = {}
    for r in reqs:
        key = r.canonical
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        # keep the stronger classification and the higher weight
        if cur.kind == "nice" and r.kind == "must":
            cur.kind = "must"
        cur.weight = max(cur.weight, r.weight)
        if len(r.skill) > len(cur.skill):
            cur.skill = r.skill          # prefer the fuller surface form
        if not cur.evidence:
            cur.evidence = r.evidence
    return list(best.values())


# --------------------------------------------------------------------------
# LLM-free fallback
# --------------------------------------------------------------------------

_REQ_HEAD_RE = re.compile(
    r"^\s*(?:minimum|basic|required|requirements?|qualifications?|what you.{0,12}ll need"
    r"|must have|you have|about you)\b", re.I | re.M)
_NICE_HEAD_RE = re.compile(
    r"^\s*(?:preferred|nice to have|bonus|desirable|plus(?:es)?|good to have"
    r"|additionally|we.{0,3}d love)\b", re.I | re.M)

_SENIORITY_RE = [
    (re.compile(r"\b(principal|distinguished|fellow)\b", re.I), "principal"),
    (re.compile(r"\b(staff|lead|tech lead|technical lead)\b", re.I), "staff"),
    (re.compile(r"\b(senior|sr\.?|sde\s*(?:ii|iii|2|3))\b", re.I), "senior"),
    (re.compile(r"\b(manager|em\b|engineering manager)\b", re.I), "manager"),
    (re.compile(r"\b(junior|jr\.?|entry.level|graduate|fresher)\b", re.I), "junior"),
    (re.compile(r"\b(intern|internship)\b", re.I), "intern"),
]

_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-|to)?\s*(\d{1,2})?\s*years?", re.I)


def heuristic_analysis(jd_text: str, title_hint: str = "",
                       company_hint: str = "") -> JDAnalysis:
    a = JDAnalysis(raw_text=jd_text, role_title=title_hint, company=company_hint)

    for rx, label in _SENIORITY_RE:
        if rx.search(title_hint) or rx.search(jd_text[:1200]):
            a.seniority = label
            break

    m = _YEARS_RE.search(jd_text)
    if m:
        a.years_experience = f"{m.group(1)}+" if not m.group(2) else f"{m.group(1)}-{m.group(2)}"

    # split the posting into a "required" region and a "preferred" region
    nice_start = None
    nm = _NICE_HEAD_RE.search(jd_text)
    if nm:
        nice_start = nm.start()
    must_text = jd_text[:nice_start] if nice_start else jd_text
    nice_text = jd_text[nice_start:] if nice_start else ""

    counts: Dict[str, int] = {}
    for canon in S.ALIASES:
        n = len(S.find_mentions(canon, jd_text))
        if n:
            counts[canon] = n

    for canon, n in counts.items():
        in_title = bool(S.mentioned_in(canon, title_hint))
        in_nice = bool(nice_text) and S.mentioned_in(canon, nice_text)
        in_must = S.mentioned_in(canon, must_text)
        kind = "must" if (in_must or in_title) else ("nice" if in_nice else "must")
        weight = min(5, 1 + n + (2 if in_title else 0))
        hit = S.find_mentions(canon, jd_text)
        ev = ""
        if hit:
            pos = hit[0].start()
            ev = jd_text[max(0, pos - 70) : pos + 90].replace("\n", " ").strip()
        # surface form as the posting spells it
        surface = hit[0].group(0) if hit else canon
        a.requirements.append(
            Requirement(skill=surface, evidence=ev, weight=weight, kind=kind)
        )

    a.requirements = _dedupe(a.requirements)
    a.ats_keywords = [r.skill for r in sorted(a.requirements, key=lambda r: -r.weight)[:25]]

    for line in jd_text.splitlines():
        ls = line.strip(" •-*\t")
        if 25 <= len(ls) <= 170 and re.match(r"^(design|build|develop|own|lead|drive|"
                                             r"collaborate|partner|mentor|deliver|"
                                             r"implement|maintain|scale|work)", ls, re.I):
            a.responsibilities.append(ls)
    a.responsibilities = a.responsibilities[:10]
    return a


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def analyse(jd_text: str, client: Optional[LLMClient] = None, *,
            title_hint: str = "", company_hint: str = "") -> JDAnalysis:
    """Structured JD analysis, LLM-assisted when possible."""
    base = heuristic_analysis(jd_text, title_hint, company_hint)
    if client is None or not client.available:
        return base

    try:
        data = client.complete_json(
            SYSTEM, PROMPT.format(jd=jd_text[:24000]),
            max_tokens=3000, temperature=0.0, label="jd_extract",
        )
    except LLMError:
        return base

    if not isinstance(data, dict):
        return base

    a = JDAnalysis(raw_text=jd_text, used_llm=True)
    a.role_title = str(data.get("role_title") or title_hint or "")
    a.company = str(data.get("company") or company_hint or "")
    a.seniority = str(data.get("seniority") or base.seniority or "unclear")
    a.years_experience = str(data.get("years_experience") or base.years_experience or "")
    a.domain = str(data.get("domain") or "")
    a.responsibilities = [str(x) for x in (data.get("responsibilities") or [])][:14]
    a.ats_keywords = [str(x) for x in (data.get("ats_keywords") or [])][:35]
    a.soft_signals = [str(x) for x in (data.get("soft_signals") or [])][:10]
    a.red_flags = [str(x) for x in (data.get("red_flags") or [])][:8]

    for kind, key in (("must", "must_have"), ("nice", "nice_to_have")):
        for item in data.get(key) or []:
            if isinstance(item, str):
                a.requirements.append(Requirement(skill=item, kind=kind))
            elif isinstance(item, dict) and item.get("skill"):
                a.requirements.append(Requirement(
                    skill=str(item["skill"]),
                    evidence=str(item.get("evidence", ""))[:300],
                    weight=int(item.get("weight") or 3),
                    kind=kind,
                ))

    # union with the heuristic pass so a vocabulary hit is never lost
    known = {r.canonical for r in a.requirements}
    for r in base.requirements:
        if r.canonical not in known and r.weight >= 3:
            a.requirements.append(r)

    a.requirements = _dedupe(a.requirements)
    if not a.ats_keywords:
        a.ats_keywords = [r.skill for r in a.must_haves()[:20]]
    return a
