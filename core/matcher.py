"""
Gap analysis: what the job asks for versus what the resume actually proves.

The distinction that matters is between *listed* and *evidenced*. A skills line
that says "Apache Kafka" gets you past a keyword filter and then falls apart in
the phone screen. A bullet that says what you built with Kafka and what it moved
is what gets you the call. So we grade every requirement on evidence depth, not
just presence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from . import skills as S
from .jd_extract import JDAnalysis, Requirement
from .latexdoc import ResumeDoc

# how we describe each level to the user
LEVEL_LABEL = {
    "strong": "Strong — proven in a bullet",
    "present": "Present — mentioned, could be stronger",
    "listed_only": "Listed only — in your skills line, no story behind it",
    "declared": "You know it — but it is nowhere on the resume",
    "missing": "Missing — no evidence at all",
}

LEVEL_ORDER = ["missing", "declared", "listed_only", "present", "strong"]

# score contribution per level
LEVEL_CREDIT = {
    "strong": 1.0,
    "present": 0.75,
    "listed_only": 0.45,
    "declared": 0.0,     # credit only after we actually write it in
    "missing": 0.0,
}


@dataclass
class Gap:
    requirement: Requirement
    level: str
    evidence: S.Evidence
    bullet_ids: List[str] = field(default_factory=list)
    adjacent: List[str] = field(default_factory=list)
    action: str = ""

    @property
    def skill(self) -> str:
        return self.requirement.skill

    @property
    def canonical(self) -> str:
        return self.requirement.canonical

    @property
    def weight(self) -> int:
        return self.requirement.weight

    @property
    def is_must(self) -> bool:
        return self.requirement.kind == "must"

    @property
    def priority(self) -> float:
        """Higher = fix this first."""
        base = self.weight * (1.6 if self.is_must else 1.0)
        deficit = 1.0 - LEVEL_CREDIT.get(self.level, 0.0)
        return round(base * deficit, 2)


@dataclass
class GapReport:
    gaps: List[Gap] = field(default_factory=list)
    declared_skills: List[str] = field(default_factory=list)
    resume_skills: List[str] = field(default_factory=list)

    def by_level(self, *levels: str) -> List[Gap]:
        want = set(levels)
        return [g for g in self.gaps if g.level in want]

    def actionable(self) -> List[Gap]:
        """Things the tailoring agent can legitimately act on, hardest first."""
        return sorted(
            [g for g in self.gaps if g.level in ("declared", "listed_only", "present")],
            key=lambda g: -g.priority,
        )

    def true_gaps(self) -> List[Gap]:
        """Things no amount of rewriting can fix — you have to go learn them."""
        return sorted([g for g in self.gaps if g.level == "missing"],
                      key=lambda g: -g.priority)

    def coverage(self) -> Dict[str, float]:
        def score(subset: List[Gap]) -> float:
            if not subset:
                return 1.0
            total = sum(g.weight for g in subset)
            got = sum(g.weight * LEVEL_CREDIT.get(g.level, 0.0) for g in subset)
            return round(got / total, 4) if total else 1.0

        musts = [g for g in self.gaps if g.is_must]
        nices = [g for g in self.gaps if not g.is_must]
        return {
            "must_have": score(musts),
            "nice_to_have": score(nices),
            "overall": round(0.75 * score(musts) + 0.25 * score(nices), 4),
        }

    def allowed_skill_vocabulary(self) -> Set[str]:
        """Everything the agent is permitted to claim.

        This is the anti-fabrication boundary: the union of what is already on
        the resume and what the user explicitly told us they know.
        """
        vocab: Set[str] = set()
        for s in self.resume_skills:
            vocab.add(S.canonical(s))
        for s in self.declared_skills:
            vocab.add(S.canonical(s))
        for g in self.gaps:
            if g.level in ("strong", "present", "listed_only", "declared"):
                vocab.add(g.canonical)
        return vocab


def _suggest_action(gap: Gap) -> str:
    lvl = gap.level
    sk = gap.skill
    if lvl == "declared":
        return (f"You know {sk} but it is invisible here. Add it to your skills line "
                f"and work it into the bullet where you actually used it.")
    if lvl == "listed_only":
        return (f"{sk} is in your skills list with nothing behind it. Rewrite one bullet "
                f"to show what you built with it and what it changed.")
    if lvl == "present":
        return (f"{sk} appears but reads thin. Add the scale or the outcome — "
                f"what it handled, what it saved, what it sped up.")
    if lvl == "missing":
        if gap.adjacent:
            near = ", ".join(gap.adjacent[:3])
            return (f"No {sk} anywhere. Closest thing you have is {near} — "
                    f"worth a weekend project before you apply, or be ready to "
                    f"bridge from {near} in the interview.")
        return (f"No {sk} anywhere, and nothing adjacent. If this is a hard "
                f"requirement, this role is a stretch — do not fake it.")
    return ""


def build_report(doc: ResumeDoc, jd: JDAnalysis,
                 declared_skills: Optional[Iterable[str]] = None) -> GapReport:
    declared = [d.strip() for d in (declared_skills or []) if d and d.strip()]
    declared_canon = {S.canonical(d) for d in declared}

    skills_text = " , ".join(doc.all_skill_values())
    report = GapReport(declared_skills=declared,
                       resume_skills=doc.all_skill_values())

    for req in jd.requirements:
        ev = S.assess(req.skill, doc.bullets, skills_text)
        if ev.level == "strong":
            level = "strong"
        elif ev.level == "mentioned":
            level = "present"
        elif ev.level == "listed_only":
            level = "listed_only"
        elif req.canonical in declared_canon:
            level = "declared"
        else:
            level = "missing"

        owned = set(doc.all_skill_values()) | set(declared)
        gap = Gap(requirement=req, level=level, evidence=ev,
                  bullet_ids=list(ev.bullet_ids),
                  adjacent=S.adjacent_to(req.skill, owned))
        gap.action = _suggest_action(gap)
        report.gaps.append(gap)

    report.gaps.sort(key=lambda g: (-g.priority, g.skill.lower()))
    return report


def rebuild_after(doc: ResumeDoc, report: GapReport, new_tex: str) -> GapReport:
    """Re-grade a report against a tailored .tex (for before/after comparison)."""
    from .latexdoc import parse as _parse

    new_doc = _parse(new_tex)
    skills_text = " , ".join(new_doc.all_skill_values())
    out = GapReport(declared_skills=report.declared_skills,
                    resume_skills=new_doc.all_skill_values())
    for g in report.gaps:
        ev = S.assess(g.skill, new_doc.bullets, skills_text)
        level = {"strong": "strong", "mentioned": "present",
                 "listed_only": "listed_only", "missing": "missing"}[ev.level]
        if level == "missing" and g.canonical in {S.canonical(d) for d in report.declared_skills}:
            level = "declared"
        ng = Gap(requirement=g.requirement, level=level, evidence=ev,
                 bullet_ids=list(ev.bullet_ids), adjacent=g.adjacent)
        ng.action = _suggest_action(ng)
        out.gaps.append(ng)
    out.gaps.sort(key=lambda g: (-g.priority, g.skill.lower()))
    return out
