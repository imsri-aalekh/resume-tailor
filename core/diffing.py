"""Word-level diffs for showing what actually changed in a bullet."""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass
from typing import List, Tuple

_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text or "") if t != ""]


@dataclass
class DiffChunk:
    kind: str          # equal | insert | delete
    text: str


def word_diff(old: str, new: str) -> List[DiffChunk]:
    a, b = tokenize(old), tokenize(new)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out: List[DiffChunk] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(DiffChunk("equal", "".join(a[i1:i2])))
        elif tag == "delete":
            out.append(DiffChunk("delete", "".join(a[i1:i2])))
        elif tag == "insert":
            out.append(DiffChunk("insert", "".join(b[j1:j2])))
        else:
            out.append(DiffChunk("delete", "".join(a[i1:i2])))
            out.append(DiffChunk("insert", "".join(b[j1:j2])))
    return [c for c in out if c.text]


def to_html(old: str, new: str) -> str:
    """Inline diff as HTML, safe for st.markdown(unsafe_allow_html=True)."""
    parts: List[str] = []
    for c in word_diff(old, new):
        t = html.escape(c.text)
        if c.kind == "equal":
            parts.append(t)
        elif c.kind == "insert":
            parts.append(
                f'<span style="background:#1d4e29;color:#b7f5c4;border-radius:3px;'
                f'padding:0 2px;">{t}</span>')
        else:
            parts.append(
                f'<span style="background:#4e1d22;color:#f5b7bd;border-radius:3px;'
                f'padding:0 2px;text-decoration:line-through;opacity:.75;">{t}</span>')
    return (f'<div style="line-height:1.7;font-size:0.92rem;">{"".join(parts)}</div>')


def unified(old_tex: str, new_tex: str, name: str = "resume.tex") -> str:
    return "".join(difflib.unified_diff(
        old_tex.splitlines(keepends=True),
        new_tex.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}", n=2,
    ))


def change_stats(old: str, new: str) -> Tuple[int, int]:
    """(words added, words removed)"""
    added = removed = 0
    for c in word_diff(old, new):
        n = len([t for t in tokenize(c.text) if t.strip()])
        if c.kind == "insert":
            added += n
        elif c.kind == "delete":
            removed += n
    return added, removed
