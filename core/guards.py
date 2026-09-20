"""
Code-enforced limits on what the tailoring agent may write.

Prompts asking a model not to exaggerate are a suggestion. These are rules.
Every rewritten bullet is checked here before it is allowed into a draft, and
anything that fails is reverted to your original text and reported back to the
critic as a violation it has to fix.

Three things we refuse to let through:

  1. **Invented skills.** A rewrite may only mention technologies already on
     your resume or ones you explicitly ticked as "I have used this".
  2. **Invented numbers.** Every digit in a rewritten bullet must already exist
     somewhere in your original resume. This is the single most common way an
     LLM will quietly lie on your behalf.
  3. **Broken LaTeX.** Unbalanced braces or new macros would stop the file
     compiling, or worse, change the template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from . import skills as S
from .latexdoc import Bullet, ResumeDoc, match_brace, strip_latex

MAX_BULLET_WORDS = 42
MIN_BULLET_WORDS = 6

# macros a bullet is allowed to contain (anything else is template drift)
ALLOWED_MACROS = {
    "textbf", "textit", "emph", "underline", "texttt", "textsc", "href", "url",
    "item", "\\", "%", "&", "$", "#", "_", "{", "}", "~", "LaTeX", "TeX",
    "hfill", "newline", "textbackslash", "ldots", "dots",
}

_NUMBER_RE = re.compile(r"\d[\d,.]*")
_MACRO_RE = re.compile(r"\\([a-zA-Z@]+|\\|[%&$#_{}~])")


@dataclass
class Violation:
    bullet_id: str
    kind: str
    message: str
    detail: str = ""


@dataclass
class ValidationResult:
    accepted: Dict[str, str] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    violations: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if not self.violations:
            return "All rewrites passed validation."
        by_kind: Dict[str, int] = {}
        for v in self.violations:
            by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        return "; ".join(f"{k}: {n}" for k, n in sorted(by_kind.items()))


# --------------------------------------------------------------------------
# number provenance
# --------------------------------------------------------------------------


def _numbers(text: str) -> Set[str]:
    out: Set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        tok = m.group(0).strip(".,")
        if tok:
            out.add(tok.replace(",", ""))
    return out


def allowed_numbers(doc: ResumeDoc) -> Set[str]:
    """Every number that legitimately exists in the original resume."""
    return _numbers(doc.plain_text())


# --------------------------------------------------------------------------
# skill provenance
# --------------------------------------------------------------------------


def skills_in(text: str) -> Set[str]:
    return {S.canonical(s) for s in S.extract_candidate_skills(text)}


# --------------------------------------------------------------------------
# LaTeX safety
# --------------------------------------------------------------------------


def latex_problems(text: str) -> List[str]:
    problems: List[str] = []

    depth = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                problems.append("unbalanced closing brace")
                break
        i += 1
    if depth > 0:
        problems.append("unclosed brace")

    for m in _MACRO_RE.finditer(text):
        name = m.group(1)
        if name not in ALLOWED_MACROS:
            problems.append(f"unexpected macro \\{name}")

    for bad, why in (("\\item", "a bullet may not contain another \\item"),
                     ("\\begin", "a bullet may not open an environment"),
                     ("\\end", "a bullet may not close an environment"),
                     ("\\section", "a bullet may not create a section")):
        if bad in text:
            problems.append(why)

    # unescaped characters that break a LaTeX build
    for ch in ("%", "&", "#", "$", "_"):
        for m in re.finditer(re.escape(ch), text):
            if m.start() == 0 or text[m.start() - 1] != "\\":
                problems.append(f"unescaped '{ch}' (write \\{ch})")
                break
    return problems


# --------------------------------------------------------------------------
# the validator
# --------------------------------------------------------------------------


def validate_rewrites(doc: ResumeDoc, rewrites: Dict[str, str],
                      allowed_vocab: Set[str],
                      extra_numbers: Optional[Set[str]] = None) -> ValidationResult:
    """Check every proposed bullet rewrite. Failures revert to the original."""
    res = ValidationResult()
    nums_ok = allowed_numbers(doc) | (extra_numbers or set())
    vocab = {S.canonical(v) for v in allowed_vocab}

    for bid, new_text in (rewrites or {}).items():
        bullet = doc.bullet(bid)
        if bullet is None:
            res.violations.append(Violation(bid, "unknown_bullet",
                                            f"No bullet {bid} exists in the resume."))
            continue
        if new_text is None or not str(new_text).strip():
            res.rejected[bid] = bullet.text
            res.violations.append(Violation(bid, "empty",
                                            "Rewrite was empty; kept the original."))
            continue

        new_text = str(new_text).strip()
        plain_new = strip_latex(new_text)
        plain_old = strip_latex(bullet.text)

        problems: List[Violation] = []

        # 1. LaTeX safety
        for p in latex_problems(new_text):
            problems.append(Violation(bid, "latex", p, new_text[:160]))

        # 2. length
        wc = len(plain_new.split())
        if wc > MAX_BULLET_WORDS:
            problems.append(Violation(bid, "too_long",
                                      f"{wc} words (limit {MAX_BULLET_WORDS}).", plain_new[:160]))
        if wc < MIN_BULLET_WORDS:
            problems.append(Violation(bid, "too_short",
                                      f"{wc} words is not a bullet.", plain_new[:160]))

        # 3. invented numbers
        new_nums = _numbers(plain_new) - _numbers(plain_old)
        invented = {n for n in new_nums if n not in nums_ok}
        if invented:
            problems.append(Violation(
                bid, "invented_metric",
                f"Introduces number(s) not found anywhere in your resume: "
                f"{', '.join(sorted(invented))}.", plain_new[:200]))

        # 4. invented skills
        new_skills = skills_in(plain_new) - skills_in(plain_old)
        unbacked = {s for s in new_skills if s not in vocab}
        if unbacked:
            problems.append(Violation(
                bid, "unbacked_skill",
                f"Claims experience you have not confirmed: "
                f"{', '.join(sorted(S.display_name(s) for s in unbacked))}.",
                plain_new[:200]))

        # 5. entity drift — a company/product name must not be swapped out
        old_caps = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", plain_old))
        new_caps = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", plain_new))
        dropped_proper = {c for c in old_caps - new_caps
                          if c not in {"Built", "Designed", "Led", "Implemented",
                                       "Migrated", "Developed", "Managed", "Owned",
                                       "Orchestrated", "Partnered", "Cut", "Unified",
                                       "Compared", "Engineered", "Deployed", "The"}}
        if len(dropped_proper) >= 2:
            problems.append(Violation(
                bid, "entity_drift",
                f"Dropped named entities from the original: "
                f"{', '.join(sorted(dropped_proper))}.", plain_new[:200]))

        if problems:
            res.rejected[bid] = bullet.text
            res.violations.extend(problems)
        else:
            res.accepted[bid] = new_text

    return res


def validate_skill_lines(doc: ResumeDoc, edits: Dict[str, str],
                         allowed_vocab: Set[str]) -> ValidationResult:
    """Skill-line edits may reorder and add confirmed skills, never invent them."""
    res = ValidationResult()
    vocab = {S.canonical(v) for v in allowed_vocab}

    for sid, new_vals in (edits or {}).items():
        line = doc.skill_line(sid)
        if line is None:
            res.violations.append(Violation(sid, "unknown_skill_line",
                                            f"No skills row {sid}."))
            continue
        new_vals = str(new_vals or "").strip()
        if not new_vals:
            res.rejected[sid] = line.raw
            continue

        problems = [Violation(sid, "latex", p, new_vals[:160])
                    for p in latex_problems(new_vals)]

        old_set = {S.canonical(v) for v in line.values()}
        new_items = [v.strip() for v in re.split(r"[,;]", strip_latex(new_vals)) if v.strip()]
        added = {S.canonical(v) for v in new_items} - old_set
        unbacked = {a for a in added if a not in vocab}
        if unbacked:
            problems.append(Violation(
                sid, "unbacked_skill",
                f"Adds unconfirmed skill(s): "
                f"{', '.join(sorted(S.display_name(u) for u in unbacked))}.",
                new_vals[:200]))

        # don't let the line silently lose skills
        lost = old_set - {S.canonical(v) for v in new_items}
        if lost:
            problems.append(Violation(
                sid, "dropped_skill",
                f"Removed skill(s) you already had: "
                f"{', '.join(sorted(S.display_name(l) for l in lost))}.",
                new_vals[:200]))

        if len(new_items) > len(line.values()) + 4:
            problems.append(Violation(sid, "too_long",
                                      "Added more than four new skills to one row."))

        if problems:
            res.rejected[sid] = line.raw
            res.violations.extend(problems)
        else:
            res.accepted[sid] = new_vals
    return res
