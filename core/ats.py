"""
ATS scoring.

Two kinds of check live here and they are not the same thing:

  * **Relevance** — does this resume answer this job description? Keyword
    coverage weighted by how badly the posting wants each thing, plus how
    deeply each one is evidenced.
  * **Parseability** — can an applicant tracking system actually read the file?
    The only honest way to test this is to compile the PDF and extract the text
    back out, which is what `parseability_from_pdf` does.

A score is a diagnostic, not a target. Anything above ~85 usually means the
resume is being written for the robot instead of the human on the other side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import skills as S
from .jd_extract import JDAnalysis
from .latexdoc import ResumeDoc
from .matcher import GapReport

_METRIC_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*%|\b\d[\d,]*\s*(?:k|m|bn|mn|lakh|crore|million|billion|x)\+?\b"
    r"|\bp\d{2}\b|\b\d+\s*(?:ms|qps|rps|tps|req/s|users|customers|services|engineers)\b"
    r"|(?:Rs\.?|INR|USD|\$|₹)\s*\d)",
    re.I,
)

_WEAK_OPENERS = re.compile(
    r"^\s*(?:worked on|helped|assisted|was responsible|responsible for|involved in|"
    r"participated in|took part|supported|handled|used|utilis|utiliz)", re.I)

_FIRST_PERSON = re.compile(r"\b(?:I|my|me|we|our)\b")

_PRONOUN_OK_CONTEXT = re.compile(r"\bI/O\b|\bIaC\b")


@dataclass
class Check:
    key: str
    label: str
    passed: bool
    detail: str = ""
    severity: str = "warn"          # info | warn | fail
    items: List[str] = field(default_factory=list)


@dataclass
class ATSScore:
    keyword_coverage: float = 0.0        # 0-1
    evidence_depth: float = 0.0
    quantification: float = 0.0
    title_alignment: float = 0.0
    format_health: float = 0.0
    total: int = 0
    checks: List[Check] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    stuffing_warning: str = ""

    def band(self) -> str:
        t = self.total
        if t >= 85:
            return "Excellent"
        if t >= 72:
            return "Strong"
        if t >= 58:
            return "Competitive"
        if t >= 42:
            return "Needs work"
        return "Weak fit"

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]


# weights must sum to 100
WEIGHTS = {
    "keyword_coverage": 38,
    "evidence_depth": 24,
    "quantification": 16,
    "title_alignment": 8,
    "format_health": 14,
}


def _quantification_rate(doc: ResumeDoc) -> float:
    exp = [b for b in doc.bullets if b.section in ("experience", "projects")]
    if not exp:
        return 0.0
    hits = sum(1 for b in exp if _METRIC_RE.search(b.plain()))
    return hits / len(exp)


def _title_alignment(doc: ResumeDoc, jd: JDAnalysis) -> float:
    if not jd.role_title:
        return 0.6
    title_words = {w for w in S.norm(jd.role_title).split() if len(w) > 2}
    stop = {"and", "the", "for", "with", "senior", "staff", "lead", "iii", "ii"}
    title_words -= stop
    if not title_words:
        return 0.6
    resume_titles = " ".join(e.title + " " + e.label for e in doc.entries
                             if e.section == "experience")
    resume_words = set(S.norm(resume_titles).split())
    if not resume_words:
        return 0.4
    overlap = len(title_words & resume_words) / len(title_words)
    return min(1.0, 0.35 + 0.65 * overlap)


def _format_checks(doc: ResumeDoc) -> List[Check]:
    checks: List[Check] = []
    text = doc.plain_text()

    # contact reachability
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", doc.source))
    has_phone = bool(re.search(r"(\+?\d[\d\s().-]{7,}\d)", text))
    checks.append(Check("contact", "Email and phone are present",
                        has_email and has_phone,
                        "An ATS that cannot find contact details may drop the record.",
                        "fail" if not has_email else "warn"))

    # section headings an ATS recognises
    norms = {s.norm for s in doc.sections}
    wanted = {"experience", "education", "skills"}
    missing = sorted(wanted - norms)
    checks.append(Check("sections", "Standard section headings",
                        not missing,
                        f"Missing recognisable heading(s): {', '.join(missing)}" if missing
                        else "Experience, Education and Skills all detected.",
                        "fail" if missing else "info", missing))

    # dates on every experience entry
    undated = [e.label for e in doc.entries
               if e.section == "experience" and not e.dates]
    checks.append(Check("dates", "Every role carries a date range",
                        not undated,
                        "Roles without parseable dates break tenure calculations." if undated
                        else "All roles have date ranges.",
                        "warn", undated))

    # first-person pronouns
    pronoun_bullets = [b.bid for b in doc.bullets
                       if _FIRST_PERSON.search(b.plain())
                       and not _PRONOUN_OK_CONTEXT.search(b.plain())]
    checks.append(Check("pronouns", "No first-person pronouns",
                        not pronoun_bullets,
                        "Resume bullets conventionally drop I/we." if pronoun_bullets
                        else "Clean.", "warn", pronoun_bullets))

    # weak openers
    weak = [b.bid for b in doc.bullets if _WEAK_OPENERS.search(b.plain())]
    checks.append(Check("weak_verbs", "Bullets open with strong verbs",
                        not weak,
                        "Bullets starting with 'worked on' or 'responsible for' read passive."
                        if weak else "All bullets open with an action verb.",
                        "warn", weak))

    # bullet length
    longs = [b.bid for b in doc.bullets if len(b.plain().split()) > 38]
    checks.append(Check("bullet_length", "Bullets stay scannable (<38 words)",
                        not longs,
                        "Over-long bullets get skimmed past." if longs
                        else "All bullets are a readable length.", "warn", longs))

    # risky layout constructs an ATS can mangle
    risky = []
    for macro, why in (
        (r"\\begin\{multicols\}", "multi-column layout"),
        (r"\\includegraphics", "images"),
        (r"\\begin\{tikzpicture\}", "TikZ graphics"),
        (r"\\fancyhead", "header/footer content"),
    ):
        if re.search(macro, doc.source):
            risky.append(why)
    checks.append(Check("layout", "No ATS-hostile layout constructs",
                        not risky,
                        f"Detected: {', '.join(risky)}. Many parsers drop these."
                        if risky else "Single-column, text-first layout.",
                        "warn", risky))
    return checks


def _stuffing_check(doc: ResumeDoc, jd: JDAnalysis) -> str:
    """Catch a resume that has been optimised into nonsense."""
    bullets = [b.plain() for b in doc.bullets if b.section in ("experience", "projects")]
    if not bullets:
        return ""
    problems: List[str] = []
    for req in jd.must_haves()[:20]:
        n = sum(len(S.find_mentions(req.skill, b)) for b in bullets)
        if n >= 5:
            problems.append(f"{S.display_name(req.skill)} ({n}x)")
    dense = [b for b in bullets
             if len(S.extract_candidate_skills(b)) >= 7 and len(b.split()) < 35]
    if problems:
        return ("Keyword repetition is high for " + ", ".join(problems[:4]) +
                ". Recruiters notice this faster than ATS rewards it.")
    if dense:
        return (f"{len(dense)} bullet(s) read as a technology list rather than an "
                f"accomplishment. Keep one or two technologies per bullet.")
    return ""


def score(doc: ResumeDoc, jd: JDAnalysis, report: GapReport,
          extra_checks: Optional[List[Check]] = None) -> ATSScore:
    cov = report.coverage()

    evidenced = [g for g in report.gaps if g.is_must]
    depth = 0.0
    if evidenced:
        total_w = sum(g.weight for g in evidenced)
        got = sum(g.weight * (1.0 if g.level == "strong" else
                              0.6 if g.level == "present" else 0.0)
                  for g in evidenced)
        depth = got / total_w if total_w else 0.0

    s = ATSScore()
    s.keyword_coverage = cov["overall"]
    s.evidence_depth = round(depth, 4)
    s.quantification = round(_quantification_rate(doc), 4)
    s.title_alignment = round(_title_alignment(doc, jd), 4)

    checks = _format_checks(doc)
    if extra_checks:
        checks.extend(extra_checks)
    s.checks = checks
    hard = sum(1 for c in checks if not c.passed and c.severity == "fail")
    soft = sum(1 for c in checks if not c.passed and c.severity == "warn")
    s.format_health = max(0.0, 1.0 - (0.34 * hard) - (0.11 * soft))

    s.breakdown = {
        k: round(getattr(s, k) * w, 2) for k, w in WEIGHTS.items()
    }
    s.total = int(round(sum(s.breakdown.values())))
    s.stuffing_warning = _stuffing_check(doc, jd)
    return s


# --------------------------------------------------------------------------
# real parseability test: compile, then read the text back out
# --------------------------------------------------------------------------


def parseability_from_pdf(pdf_bytes: bytes, doc: ResumeDoc) -> List[Check]:
    """Extract text from the compiled PDF and confirm the content survived."""
    checks: List[Check] = []
    try:
        import pdfplumber
        import io

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = len(pdf.pages)
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as exc:
        return [Check("pdf_parse", "PDF text extraction", False,
                      f"Could not read the compiled PDF back ({type(exc).__name__}).",
                      "warn")]

    words = len(text.split())
    checks.append(Check("pdf_text", "PDF contains selectable text",
                        words > 120,
                        f"{words} words extracted." if words > 120 else
                        "Almost no text came back — an ATS would see a blank page.",
                        "fail" if words <= 120 else "info"))

    checks.append(Check("pdf_pages", "Page count is recruiter-friendly",
                        pages <= 2,
                        f"{pages} page(s)." + ("" if pages <= 2 else
                        " Two pages is the practical ceiling for most reviewers."),
                        "warn" if pages > 2 else "info"))

    # Did every bullet actually make it into the extracted text?
    # Two different failures hide here and they have very different severity:
    #   * content genuinely missing  -> an ATS sees nothing, this is fatal
    #   * content present but with the spaces dropped between words -> most
    #     parsers cope, a minority tokenise it into gibberish
    # so we test them separately.
    spaced = re.sub(r"\s+", " ", text.lower())
    squashed = re.sub(r"\s+", "", text.lower())

    lost: List[str] = []
    unspaced: List[str] = []
    for b in doc.bullets:
        plain = b.plain().lower()
        probe_spaced = " ".join(re.sub(r"\s+", " ", plain).split()[:6])
        if len(probe_spaced) <= 12:
            continue
        if probe_spaced in spaced:
            continue
        probe_squashed = re.sub(r"\s+", "", probe_spaced)
        if probe_squashed and probe_squashed in squashed:
            unspaced.append(b.bid)
        else:
            lost.append(b.bid)

    checks.append(Check("pdf_bullets", "Every bullet survives text extraction",
                        not lost,
                        f"{len(lost)} bullet(s) did not appear in the extracted text at "
                        f"all — an ATS would not see them." if lost else
                        "All bullet content is present in the extracted text.",
                        "fail", lost))

    checks.append(Check("pdf_spacing", "Words stay separated when extracted",
                        not unspaced,
                        f"{len(unspaced)} line(s) extract with the spaces between words "
                        f"dropped (e.g. 'ledtheintegration'). Most parsers recover, but "
                        f"strict ones will not. Usually caused by justified text or the "
                        f"font's kerning — switching the PDF engine often fixes it."
                        if unspaced else "Word spacing survives extraction.",
                        "warn", unspaced))
    return checks
