"""
Span-preserving LaTeX resume model.

The core guarantee of this module: we NEVER regenerate your resume.
We locate the exact character spans of editable *content* (bullet text,
skill values, the headline/summary) inside your original .tex, and rewrite
only those spans. Every byte of your preamble, document class, macros,
spacing, and section scaffolding is carried through untouched.

`render(doc, {})` is guaranteed byte-identical to the original source.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# low-level LaTeX scanning helpers
# --------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)


def strip_comments(tex: str) -> str:
    """Blank out comments (keeping length identical so offsets stay valid)."""
    def _blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    return _COMMENT_RE.sub(_blank, tex)


def match_brace(text: str, open_idx: int) -> int:
    """Given index of a '{', return index just past the matching '}'.

    Handles nesting and backslash-escaped braces. Returns -1 if unbalanced.
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        return -1
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def read_group(text: str, idx: int) -> Tuple[Optional[str], int]:
    """Read a {...} group starting at/after idx (skipping whitespace).

    Returns (inner_text, index_after_group) or (None, idx) if no group.
    """
    j = idx
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    if j >= len(text) or text[j] != "{":
        return None, idx
    end = match_brace(text, j)
    if end == -1:
        return None, idx
    return text[j + 1 : end - 1], end


def strip_latex(text: str) -> str:
    """Rough plain-text rendering of a LaTeX fragment, for matching and search."""
    s = text
    s = re.sub(r"\\(?:href|url)\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:textbf|textit|emph|underline|texttt|textsc|mbox|bf|it)\s*\{", "{", s)
    s = re.sub(r"\\[a-zA-Z@]+\s*\*?", " ", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\\\", " ").replace("~", " ")
    s = re.sub(r"\\[&%$#_]", lambda m: m.group(0)[1], s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# --------------------------------------------------------------------------
# document model
# --------------------------------------------------------------------------


@dataclass
class Span:
    """An editable region of the original source."""

    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass
class Bullet:
    bid: str
    span: Span
    raw: str                 # exact original slice
    text: str                # trimmed version we hand to the model
    lead_ws: str             # whitespace we must re-attach on write
    trail_ws: str
    section: str
    entry_key: str
    entry_label: str
    wrapper: str = "item"    # "item" | "macro:<name>"

    def plain(self) -> str:
        return strip_latex(self.text)


@dataclass
class Entry:
    key: str
    section: str
    label: str               # "Company — Title" best effort
    org: str = ""
    title: str = ""
    dates: str = ""
    location: str = ""
    start: int = 0
    end: int = 0
    bullet_ids: List[str] = field(default_factory=list)


@dataclass
class SkillLine:
    sid: str
    label: str
    span: Span               # span of the *values* only
    raw: str
    section: str

    def values(self) -> List[str]:
        """Split on commas that separate skills — not the ones inside a
        parenthesised gloss like "Python (Pandas, Numpy)", which names one
        skill, not three."""
        raw = strip_latex(self.raw).lstrip(":-").strip()
        out: List[str] = []
        buf: List[str] = []
        depth = 0
        for ch in raw:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch in ",;" and depth == 0:
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        out.append("".join(buf))
        return [v.strip(" :") for v in out if v.strip(" :")]


@dataclass
class Section:
    name: str
    norm: str
    start: int
    end: int
    body_start: int
    body_end: int


@dataclass
class ResumeDoc:
    source: str
    sections: List[Section] = field(default_factory=list)
    entries: List[Entry] = field(default_factory=list)
    bullets: List[Bullet] = field(default_factory=list)
    skill_lines: List[SkillLine] = field(default_factory=list)
    doc_class: str = ""
    engine_hint: str = "pdflatex"
    warnings: List[str] = field(default_factory=list)
    # Custom \newcommand heading macros discovered in the preamble.
    heading_macros: List[str] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------
    def bullet(self, bid: str) -> Optional[Bullet]:
        for b in self.bullets:
            if b.bid == bid:
                return b
        return None

    def skill_line(self, sid: str) -> Optional[SkillLine]:
        for s in self.skill_lines:
            if s.sid == sid:
                return s
        return None

    def entry(self, key: str) -> Optional[Entry]:
        for e in self.entries:
            if e.key == key:
                return e
        return None

    def section_of(self, pos: int) -> str:
        for s in self.sections:
            if s.body_start <= pos < s.body_end:
                return s.norm
        return ""

    # -- text views ------------------------------------------------------
    def plain_text(self) -> str:
        """Plain-text projection of the whole resume (for keyword search)."""
        return strip_latex(strip_comments(self.source))

    def all_skill_values(self) -> List[str]:
        out: List[str] = []
        for s in self.skill_lines:
            out.extend(s.values())
        return out

    def fingerprint(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# section detection
# --------------------------------------------------------------------------

_SECTION_PATTERNS = [
    # \begin{rSection}{EXPERIENCE}  (openfont / sb2nov family)
    re.compile(r"\\begin\s*\{(rSection|cvsection|resumeSection)\}\s*(?:\[[^\]]*\])?\s*\{", re.I),
    # \section{Experience}, \section*{...}, \cvsection{...}, \resumeSection{...}
    re.compile(r"\\(section\*?|cvsection|resheading|resumeSection|headingsection)\s*\{", re.I),
]

_SECTION_ALIASES = {
    "experience": ["experience", "work experience", "professional experience",
                   "employment", "work history", "career"],
    "education": ["education", "academics", "academic background", "qualifications"],
    "skills": ["skills", "technical skills", "technologies", "skills summary",
               "technical proficiencies", "core competencies", "tech stack"],
    "projects": ["projects", "personal projects", "academic projects",
                 "selected projects", "project work"],
    "summary": ["summary", "profile", "objective", "professional summary",
                "about", "career objective"],
    "certifications": ["certifications", "certificates", "courses", "licenses"],
    "achievements": ["achievements", "awards", "honors", "accomplishments",
                     "extra curricular", "extracurricular", "activities"],
    "publications": ["publications", "papers", "research"],
}


def normalize_section(name: str) -> str:
    clean = strip_latex(name).strip().lower()
    clean = re.sub(r"[^a-z ]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    for norm, aliases in _SECTION_ALIASES.items():
        for a in aliases:
            if clean == a or clean.startswith(a) or a in clean:
                return norm
    return clean or "other"


# A \newcommand whose body both takes an argument and *looks like* a heading:
# bold or large text, or a rule. Plenty of resumes roll their own heading macro
# instead of using \section, and a fixed list of macro names cannot keep up
# with them.
_NEWCOMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\s*\*?\s*"
    r"\{?\s*\\([A-Za-z@]+)\s*\}?\s*\[\s*1\s*\]\s*(?:\[[^\]]*\])?\s*\{"
)
# Deliberately excludes \vspace: nearly every resume macro nudges vertical
# space, including the bullet macros, so it says nothing about being a heading.
_HEADING_BODY_RE = re.compile(
    r"\\(?:textbf|bfseries|section|Large|large|LARGE|scshape|uppercase|MakeUppercase"
    r"|underline|hrule|rule|titlerule|sectionfont)\b"
)
# A macro that emits a list item is a bullet macro, whatever else it does.
_ITEMISH_RE = re.compile(r"\\(?:item|resumeItem|begin\s*\{\s*itemize)\b")


def _heading_macros(full: str) -> List[str]:
    r"""Names of single-argument macros defined in the preamble that behave like
    section headings, e.g. ``\newcommand{\sectionhead}[1]{\textbf{#1}}``."""
    found: List[str] = []
    for m in _NEWCOMMAND_RE.finditer(full):
        name = m.group(1)
        if name in {"item", "bf", "it"} or "item" in name.lower():
            continue
        close = match_brace(full, m.end() - 1)
        if close == -1:
            continue
        body = full[m.end() : close - 1]
        if "#1" not in body or _ITEMISH_RE.search(body):
            continue
        if _HEADING_BODY_RE.search(body):
            found.append(name)
    return found


def _find_sections(source: str, scan: str, extra_macros: Optional[List[str]] = None) -> List[Section]:
    hits: List[Tuple[int, int, str]] = []
    patterns = list(_SECTION_PATTERNS)
    if extra_macros:
        patterns.append(re.compile(
            r"\\(" + "|".join(re.escape(n) for n in sorted(set(extra_macros))) + r")\s*\{"
        ))
    for pat in patterns:
        for m in pat.finditer(scan):
            brace = m.end() - 1
            close = match_brace(scan, brace)
            if close == -1:
                continue
            name = scan[brace + 1 : close - 1]
            hits.append((m.start(), close, name))
    hits.sort(key=lambda h: h[0])

    # de-duplicate overlapping matches (keep the earliest starting one)
    deduped: List[Tuple[int, int, str]] = []
    for h in hits:
        if deduped and h[0] < deduped[-1][1]:
            continue
        deduped.append(h)

    sections: List[Section] = []
    for i, (start, head_end, name) in enumerate(deduped):
        body_end = deduped[i + 1][0] if i + 1 < len(deduped) else len(scan)
        # stop at \end{document}
        endoc = scan.find("\\end{document}", head_end)
        if endoc != -1:
            body_end = min(body_end, endoc)
        sections.append(
            Section(
                name=strip_latex(name).strip(),
                norm=normalize_section(name),
                start=start,
                end=body_end,
                body_start=head_end,
                body_end=body_end,
            )
        )
    return sections


# --------------------------------------------------------------------------
# bullet detection
# --------------------------------------------------------------------------

# Macro-style bullets used by popular resume classes
_MACRO_BULLET_RE = re.compile(
    r"\\(resumeItem|cvitem|resumeSubItem|cvItem|achievement)\s*(?:\{[^{}]{0,80}\})?\s*\{"
)

_ITEM_RE = re.compile(r"\\item\b\s*(?:\[[^\]]*\])?")

_BULLET_STOP_RE = re.compile(
    r"\\item\b|\\end\s*\{\s*(itemize|enumerate|description|rSubsection|rSection)\s*\}"
    r"|\\resumeItemListEnd|\\resumeSubHeadingListEnd|\\resumeItem\b|\\cvitem\b"
)


def _find_bullets(source: str, scan: str, sections: List[Section]) -> List[Bullet]:
    bullets: List[Bullet] = []
    claimed: List[Tuple[int, int]] = []

    # A \item inside the skills section is a skills row, not an experience
    # bullet. Leaving it to _find_skill_lines keeps one span under one editor —
    # otherwise both a bullet edit and a skill edit can target the same offsets
    # and render() rejects the overlapping patch.
    for _sec in sections:
        if _sec.norm == "skills":
            claimed.append((_sec.body_start, _sec.body_end))

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in claimed)

    # 1) macro-style bullets: \resumeItem{ ... }
    for m in _MACRO_BULLET_RE.finditer(scan):
        brace = m.end() - 1
        close = match_brace(scan, brace)
        if close == -1:
            continue
        inner_start, inner_end = brace + 1, close - 1
        if overlaps(m.start(), close):
            continue
        claimed.append((m.start(), close))
        raw = source[inner_start:inner_end]
        bullets.append(
            _mk_bullet(source, sections, inner_start, inner_end, raw,
                       wrapper=f"macro:{m.group(1)}")
        )

    # 2) classic \item bullets
    for m in _ITEM_RE.finditer(scan):
        if overlaps(m.start(), m.end()):
            continue
        text_start = m.end()
        stop = _BULLET_STOP_RE.search(scan, text_start)
        text_end = stop.start() if stop else len(scan)
        # don't run past the end of the document
        endoc = scan.find("\\end{document}", text_start)
        if endoc != -1:
            text_end = min(text_end, endoc)
        if text_end <= text_start:
            continue
        if overlaps(text_start, text_end):
            continue
        claimed.append((m.start(), text_end))
        raw = source[text_start:text_end]
        if not strip_latex(raw):
            continue
        bullets.append(_mk_bullet(source, sections, text_start, text_end, raw, wrapper="item"))

    bullets.sort(key=lambda b: b.span.start)
    for i, b in enumerate(bullets, 1):
        b.bid = f"b{i:03d}"
    return bullets


def _mk_bullet(source: str, sections: List[Section], start: int, end: int,
               raw: str, wrapper: str) -> Bullet:
    stripped = raw.strip()
    lead_len = len(raw) - len(raw.lstrip())
    trail_len = len(raw) - len(raw.rstrip())
    section = ""
    for s in sections:
        if s.body_start <= start < s.body_end:
            section = s.norm
            break
    return Bullet(
        bid="",
        span=Span(start, end),
        raw=raw,
        text=stripped,
        lead_ws=raw[:lead_len],
        trail_ws=raw[len(raw) - trail_len :] if trail_len else "",
        section=section,
        entry_key="",
        entry_label="",
        wrapper=wrapper,
    )


# --------------------------------------------------------------------------
# entry (job / project) detection
# --------------------------------------------------------------------------

_ENTRY_PATTERNS = [
    # \resumeSubheading{Company}{Location}{Title}{Dates}
    re.compile(r"\\(resumeSubheading|resumeProjectHeading|cventry|resumeSubHeading)\s*\{"),
    # \begin{rSubsection}{Title}{Dates}{Org}{Location}
    re.compile(r"\\begin\s*\{(rSubsection)\}\s*\{"),
]

_DATE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{4}"
    r"|\b(?:19|20)\d{2}\b)"
    r"\s*(?:-|--|–|—|to|until)?\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{4}"
    r"|\b(?:19|20)\d{2}\b|Present|Current|Now)?",
    re.I,
)


def _find_entries(source: str, scan: str, sections: List[Section],
                  bullets: List[Bullet]) -> List[Entry]:
    entries: List[Entry] = []

    # (a) macro-based entries
    for pat in _ENTRY_PATTERNS:
        for m in pat.finditer(scan):
            idx = m.end() - 1
            groups: List[str] = []
            cursor = idx
            for _ in range(4):
                inner, nxt = read_group(scan, cursor)
                if inner is None:
                    break
                groups.append(strip_latex(inner))
                cursor = nxt
            if not groups:
                continue
            entries.append(
                Entry(key="", section="", label=" — ".join(g for g in groups[:2] if g),
                      org=groups[0] if groups else "",
                      title=groups[2] if len(groups) > 2 else (groups[1] if len(groups) > 1 else ""),
                      dates=next((g for g in groups if _DATE_RE.search(g)), ""),
                      start=m.start(), end=cursor)
            )

    # (b) plain-LaTeX entries: a line with \textbf{...} and a \hfill date
    if not entries:
        for sec in sections:
            if sec.norm not in {"experience", "projects", "education"}:
                continue
            body = scan[sec.body_start : sec.body_end]
            for lm in re.finditer(r"[^\n]*\\hfill[^\n]*", body):
                line = lm.group(0)
                if not _DATE_RE.search(strip_latex(line)):
                    continue
                abs_start = sec.body_start + lm.start()
                parts = [strip_latex(p) for p in line.split("\\hfill")]
                left = parts[0].strip(" ,")
                right = parts[-1].strip(" ,\\") if len(parts) > 1 else ""
                entries.append(
                    Entry(key="", section=sec.norm,
                          label=left or right, org=left, title=left,
                          dates=right, start=abs_start, end=abs_start + len(line))
                )

    # (c) a bold run at the start of a line, with or without dates. Plenty of
    # plain `article` resumes separate org from dates with \quad | \quad and
    # end the line with \\, so there is no \hfill for (b) to find. Project
    # headers often carry no date at all.
    if not entries:
        for sec in sections:
            if sec.norm not in {"experience", "projects", "education"}:
                continue
            body = scan[sec.body_start : sec.body_end]
            for lm in re.finditer(r"^[ \t]*\\(?:textbf|textsc|bf)\s*\{[^\n]*$",
                                  body, re.MULTILINE):
                line = lm.group(0)
                plain = strip_latex(line).strip()
                if len(plain) < 3:
                    continue
                abs_start = sec.body_start + lm.start()
                dates = ""
                dm = _DATE_RE.search(plain)
                if dm:
                    dates = plain[dm.start():].strip(" ,|\\")
                # everything left of the first separator is the organisation
                head = re.split(r"\||\u2014|--|\s{2,}", plain)[0].strip(" ,:|")
                entries.append(
                    Entry(key="", section=sec.norm, label=head or plain[:60],
                          org=head, title=head, dates=dates,
                          start=abs_start, end=abs_start + len(line))
                )

    entries.sort(key=lambda e: e.start)
    # assign section + key, then attach bullets to the nearest preceding entry
    for i, e in enumerate(entries, 1):
        e.key = f"e{i:03d}"
        if not e.section:
            for s in sections:
                if s.body_start <= e.start < s.body_end:
                    e.section = s.norm
                    break

    for b in bullets:
        owner: Optional[Entry] = None
        for e in entries:
            if e.start <= b.span.start:
                owner = e
            else:
                break
        if owner is not None and owner.section == b.section:
            b.entry_key = owner.key
            b.entry_label = owner.label
            owner.bullet_ids.append(b.bid)
    return entries


# --------------------------------------------------------------------------
# skills detection
# --------------------------------------------------------------------------

# A tabular skills row: "Label & value, value \\".  The label may itself contain
# escaped ampersands ("Cloud \& Tools"), so we only break on an *unescaped* &.
_SKILL_ROW_TABULAR = re.compile(
    r"^(?P<label>(?:[^&\n\\]|\\[&%$#_]|\\[a-zA-Z@]+)+?)(?<!\\)&"
    r"(?P<vals>[^\n]*?)(?P<term>\\\\|$)",
    re.MULTILINE,
)
# Handles both "\\textbf{Label}: a, b" and "\\textbf{Label}{: a, b}".
_SKILL_ROW_BOLD = re.compile(
    r"\\(?:textbf|textit|bf)\s*\{(?P<label>[^{}]{2,40})\}"
    r"\s*\{?\s*[:\-]?\s*(?P<vals>[^\n{}]*?)\s*\}?\s*(?=\\\\|\n|$)"
)
# "Languages and Tools: Base SAS, ..." — optionally behind an \item, which is
# how most plain `article` resumes lay their skills section out.
_SKILL_ROW_ITEM = re.compile(
    r"^\s*(?:\\item\b\s*(?:\[[^\]]*\])?\s*)?"
    r"(?P<label>[A-Z][A-Za-z /&+#.-]{2,40}?)\s*:\s*(?P<vals>[^\n]+)$",
    re.MULTILINE,
)


def _find_skill_lines(source: str, scan: str, sections: List[Section]) -> List[SkillLine]:
    lines: List[SkillLine] = []
    seen: List[Tuple[int, int]] = []

    def claim(a: int, b: int) -> bool:
        if any(not (b <= s or a >= e) for s, e in seen):
            return False
        seen.append((a, b))
        return True

    for sec in sections:
        if sec.norm != "skills":
            continue
        body_a, body_b = sec.body_start, sec.body_end
        body = scan[body_a:body_b]
        for pat in (_SKILL_ROW_TABULAR, _SKILL_ROW_BOLD, _SKILL_ROW_ITEM):
            for m in pat.finditer(body):
                a = body_a + m.start("vals")
                b = body_a + m.end("vals")
                label = strip_latex(m.group("label")).strip(" :&")
                vals = source[a:b]
                if not strip_latex(vals) or len(strip_latex(vals)) < 2:
                    continue
                if not claim(a, b):
                    continue
                lines.append(SkillLine(sid="", label=label or "Skills",
                                       span=Span(a, b), raw=vals, section=sec.norm))
            if lines:
                break  # first pattern that works for this section wins

    lines.sort(key=lambda s: s.span.start)
    for i, s in enumerate(lines, 1):
        s.sid = f"s{i:03d}"
    return lines


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

_DOCCLASS_RE = re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")


def _mask_preamble(scan: str) -> str:
    """Blank everything before \\begin{document}, preserving offsets.

    Resume classes define their bullet macros in the preamble
    (``\\newcommand{\\resumeItem}[1]{\\item ...}``). Without this, those macro
    *definitions* get picked up as bullets and the agent tries to rewrite them.
    """
    m = re.search(r"\\begin\s*\{document\}", scan)
    if not m:
        return scan
    return " " * m.end() + scan[m.end():]


def parse(tex: str) -> ResumeDoc:
    scan = _mask_preamble(strip_comments(tex))
    doc = ResumeDoc(source=tex)

    full = strip_comments(tex)
    m = _DOCCLASS_RE.search(full)
    if m:
        doc.doc_class = m.group(1).strip()

    if re.search(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{fontspec\}", full) or \
       re.search(r"\\setmainfont", full):
        doc.engine_hint = "xelatex"

    # Heading macros are declared in the preamble, which `scan` has blanked out,
    # so they have to be read from the unmasked source.
    doc.heading_macros = _heading_macros(full)
    doc.sections = _find_sections(tex, scan, doc.heading_macros)
    doc.bullets = _find_bullets(tex, scan, doc.sections)
    doc.entries = _find_entries(tex, scan, doc.sections, doc.bullets)
    doc.skill_lines = _find_skill_lines(tex, scan, doc.sections)

    if not doc.sections:
        doc.warnings.append(
            "No sections detected — the tailoring agent will still work on bullets, "
            "but section-aware checks are disabled."
        )
    if not doc.bullets:
        doc.warnings.append(
            "No bullet points detected. Check that your resume uses \\item or a "
            "\\resumeItem{...}-style macro."
        )
    if not doc.skill_lines:
        doc.warnings.append(
            "No skills section detected — keyword injection into the skills list is disabled."
        )
    return doc


def render(doc: ResumeDoc, bullet_edits: Optional[Dict[str, str]] = None,
           skill_edits: Optional[Dict[str, str]] = None) -> str:
    """Rebuild the .tex applying only the given span replacements.

    With no edits this returns the original source byte-for-byte.
    """
    bullet_edits = bullet_edits or {}
    skill_edits = skill_edits or {}

    patches: List[Tuple[int, int, str]] = []
    for bid, new_text in bullet_edits.items():
        b = doc.bullet(bid)
        if b is None or new_text is None:
            continue
        if new_text.strip() == b.text.strip():
            continue
        patches.append((b.span.start, b.span.end, b.lead_ws + new_text.strip() + b.trail_ws))
    for sid, new_vals in skill_edits.items():
        s = doc.skill_line(sid)
        if s is None or new_vals is None:
            continue
        if new_vals.strip() == s.raw.strip():
            continue
        lead = s.raw[: len(s.raw) - len(s.raw.lstrip())]
        trail = s.raw[len(s.raw.rstrip()) :]
        patches.append((s.span.start, s.span.end, lead + new_vals.strip() + trail))

    patches.sort(key=lambda p: p[0], reverse=True)
    out = doc.source
    last_start = len(out) + 1
    for start, end, text in patches:
        if end > last_start:
            raise ValueError(f"overlapping patch at {start}:{end}")
        out = out[:start] + text + out[end:]
        last_start = start
    return out


def verify_template_integrity(doc: ResumeDoc, rendered: str,
                              bullet_edits: Dict[str, str],
                              skill_edits: Dict[str, str]) -> List[str]:
    """Confirm nothing outside the editable spans moved.

    Works by reverting the edits and checking we land back on the original.
    """
    problems: List[str] = []
    expected = render(doc, bullet_edits, skill_edits)
    if expected != rendered:
        problems.append("Rendered output does not match the span-patched source.")

    # structural invariants
    for macro in ("\\documentclass", "\\begin{document}", "\\end{document}"):
        if doc.source.count(macro) != rendered.count(macro):
            problems.append(f"Structural macro count changed for {macro}.")
    for env in ("itemize", "rSection", "tabular", "enumerate"):
        if doc.source.count(f"\\begin{{{env}}}") != rendered.count(f"\\begin{{{env}}}"):
            problems.append(f"Environment count changed for {env}.")
        if doc.source.count(f"\\end{{{env}}}") != rendered.count(f"\\end{{{env}}}"):
            problems.append(f"Environment count changed for {env} (end).")

    preamble_end = doc.source.find("\\begin{document}")
    if preamble_end != -1:
        if doc.source[:preamble_end] != rendered[: rendered.find("\\begin{document}")]:
            problems.append("Preamble was modified — template must stay untouched.")
    return problems
