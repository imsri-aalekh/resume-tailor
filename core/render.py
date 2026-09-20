"""
Compile the tailored .tex to PDF.

Streamlit Community Cloud has no LaTeX toolchain and installing one through
`packages.txt` reliably blows the build budget, so PDF output is treated as a
bonus, never a dependency. The ladder:

  1. a local engine if one exists (tectonic, then latexmk/xelatex/pdflatex)
  2. a remote compile service, if the user opts in
  3. no PDF — the .tex download and the Overleaf hand-off still work

The .tex file is always the real deliverable. Nothing here can fail in a way
that costs the user their tailored resume.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ENGINES = [
    ("tectonic", ["tectonic", "--keep-logs", "--synctex=0", "-o", "{outdir}", "{tex}"]),
    ("latexmk-xe", ["latexmk", "-xelatex", "-interaction=nonstopmode",
                    "-halt-on-error", "-outdir={outdir}", "{tex}"]),
    ("latexmk-pdf", ["latexmk", "-pdf", "-interaction=nonstopmode",
                     "-halt-on-error", "-outdir={outdir}", "{tex}"]),
    ("xelatex", ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory={outdir}", "{tex}"]),
    ("pdflatex", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                  "-output-directory={outdir}", "{tex}"]),
]

_ERROR_RE = re.compile(r"^(?:!|.*?:\d+:)\s*(.+)$", re.M)


@dataclass
class RenderResult:
    pdf: Optional[bytes] = None
    engine: str = ""
    log: str = ""
    errors: List[str] = field(default_factory=list)
    ok: bool = False
    missing_class: str = ""

    def friendly_error(self) -> str:
        if self.missing_class:
            return (
                f"LaTeX could not find `{self.missing_class}.cls` — that file is part "
                f"of your resume template and lives next to your .tex. Upload it in "
                f"the sidebar and the preview will build. Your tailored .tex is "
                f"already correct and will compile in Overleaf as-is."
            )
        if not available_engines():
            return (
                "No LaTeX engine is installed on this server, so the PDF preview is "
                "off. Download the tailored .tex and compile it in Overleaf — it uses "
                "your original template unchanged."
            )
        if self.errors:
            return "LaTeX reported: " + "; ".join(self.errors[:3])
        return "The PDF did not build. See the log below."


def available_engines() -> List[str]:
    return [name for name, cmd in ENGINES if shutil.which(cmd[0])]


def _extract_errors(log: str) -> Tuple[List[str], str]:
    errors: List[str] = []
    missing_cls = ""
    m = re.search(r"File `([^']+)\.cls' not found", log) or \
        re.search(r"LaTeX Error: File `([^']+)\.cls' not found", log)
    if m:
        missing_cls = m.group(1)
    for line in log.splitlines():
        line = line.strip()
        if line.startswith("!"):
            msg = line.lstrip("! ").strip()
            if msg and msg not in errors:
                errors.append(msg[:200])
    return errors[:8], missing_cls


def compile_tex(tex: str, support_files: Optional[Dict[str, bytes]] = None,
                timeout: int = 120, engine_hint: str = "") -> RenderResult:
    """Compile `tex` to PDF using whichever engine is available."""
    engines = [e for e in ENGINES if shutil.which(e[1][0])]
    if engine_hint:
        engines.sort(key=lambda e: 0 if engine_hint in e[0] else 1)
    if not engines:
        return RenderResult(ok=False, log="No LaTeX engine found on this machine.")

    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, "resume.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tex)
        for name, blob in (support_files or {}).items():
            safe = os.path.basename(name)
            with open(os.path.join(tmp, safe), "wb") as fh:
                fh.write(blob)

        last = RenderResult()
        for name, template in engines:
            cmd = [c.format(outdir=tmp, tex=tex_path) for c in template]
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True,
                                      timeout=timeout, text=True,
                                      env={**os.environ, "TEXMFOUTPUT": tmp})
                log = (proc.stdout or "") + "\n" + (proc.stderr or "")
            except subprocess.TimeoutExpired:
                last = RenderResult(ok=False, engine=name,
                                    log=f"{name} timed out after {timeout}s.")
                continue
            except OSError as exc:
                last = RenderResult(ok=False, engine=name, log=str(exc))
                continue

            pdf_path = os.path.join(tmp, "resume.pdf")
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 900:
                with open(pdf_path, "rb") as fh:
                    return RenderResult(pdf=fh.read(), engine=name, log=log[-6000:], ok=True)

            errors, missing = _extract_errors(log)
            last = RenderResult(ok=False, engine=name, log=log[-6000:],
                                errors=errors, missing_class=missing)
            if missing:
                break   # a missing class will fail on every engine
        return last


# --------------------------------------------------------------------------
# a minimal stand-in class, for preview only
# --------------------------------------------------------------------------

SHIM_NOTICE = (
    "Preview rendered with a generic stand-in class because the real template "
    "class file was not supplied. Spacing and fonts will differ from your actual "
    "resume; the text content is exact."
)


def shim_class(class_name: str) -> str:
    """A tiny class implementing the common resume macros, for preview only.

    Deliberately loads almost nothing: the user's own .tex will load geometry,
    hyperref, xcolor and friends with its own options, and a class that loads
    them first causes an "Option clash" that kills the build. Margins are set
    with raw dimensions so a later \geometry call simply wins.
    """
    body = r"""
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{__CLASSNAME__}[2024/01/01 Preview shim]
\LoadClass[11pt,a4paper]{article}
\RequirePackage{enumitem}
\RequirePackage{array}
\RequirePackage{etoolbox}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{0pt}
\setlength{\oddsidemargin}{-0.35in}
\setlength{\evensidemargin}{-0.35in}
\setlength{\textwidth}{7.2in}
\setlength{\topmargin}{-0.7in}
\setlength{\textheight}{10.3in}
\setlength{\headsep}{0pt}
\setlength{\headheight}{0pt}

\AfterEndPreamble{%
  \@ifundefined{href}{\newcommand{\href}[2]{#2}}{}%
  \@ifundefined{url}{\newcommand{\url}[1]{\texttt{#1}}}{}%
}

\def\@sname{}
\newcommand{\name}[1]{\gdef\@sname{#1}}
\newcommand{\@addresses}{}
\newcommand{\address}[1]{%
  \expandafter\gdef\expandafter\@addresses\expandafter{\@addresses #1\\}}

\AfterEndPreamble{%
  \begin{center}
    {\LARGE\bfseries \@sname}\\[3pt]
    {\footnotesize \begin{tabular}{c}\@addresses\end{tabular}}
  \end{center}
  \vspace{-4pt}
}

\newenvironment{rSection}[1]{%
  \vspace{6pt}
  {\large\bfseries\MakeUppercase{#1}}
  \vspace{-4pt}
  \hrule height 0.6pt
  \vspace{4pt}
  \setlist[itemize]{leftmargin=1.5em,itemsep=1pt,topsep=2pt,parsep=0pt}
}{\vspace{3pt}}

\newenvironment{rSubsection}[4]{%
  {\bfseries #1} \hfill {#2}\\
  {\itshape #3} \hfill {\itshape #4}
  \begin{itemize}
}{\end{itemize}}

\newcommand{\resumeItem}[1]{\item #1}
\newcommand{\resumeSubheading}[4]{%
  \vspace{2pt}{\bfseries #1} \hfill {#2}\\{\itshape #3} \hfill {\itshape #4}\par}
"""
    return body.replace("__CLASSNAME__", class_name)
