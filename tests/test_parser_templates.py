"""
The parser must survive templates it has never seen.

People change resume templates. If the parser only works on one .cls the whole
tool is a toy. These fixtures cover the three shapes that account for most
LaTeX resumes in the wild: a custom class with rSection environments, the
macro-heavy Jake Gutierrez family, and plain article + itemize.

The contract every template must satisfy:
  * render(doc, {}, {}) is byte-identical to the source
  * no macro *definition* from the preamble is mistaken for a bullet
  * real bullets land in the right section
  * skill rows parse into clean values with no LaTeX debris
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.latexdoc import parse, render

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


CASES = [
    # (path, expected editable bullets, expected skill rows, must-not-appear)
    (os.path.join(ROOT, "samples", "aalekh_resume.tex"), 18, 5, []),
    (os.path.join(ROOT, "tests", "fixtures", "jake_style.tex"), 3, 3,
     ["vspace", "#1", "resumeItemListStart"]),
    (os.path.join(ROOT, "tests", "fixtures", "plain_style.tex"), 3, 2, []),
]


def main():
    for path, n_editable, n_skills, forbidden in CASES:
        name = os.path.basename(path)
        print(f"\n== {name} ==")
        src = open(path, encoding="utf-8").read()
        doc = parse(src)

        check(f"{name}: byte-identical round-trip", render(doc, {}, {}) == src)

        editable = [b for b in doc.bullets if b.section in ("experience", "projects")]
        check(f"{name}: editable bullet count", len(editable) == n_editable,
              f"got {len(editable)}, want {n_editable}")

        check(f"{name}: skill rows parsed", len(doc.skill_lines) == n_skills,
              f"got {len(doc.skill_lines)}, want {n_skills}")

        # no preamble macro bodies leaked into the bullet list
        leaked = [b.bid for b in doc.bullets
                  if any(tok in b.text for tok in forbidden)]
        check(f"{name}: no preamble macros treated as bullets", not leaked, str(leaked))

        # skill values must be clean strings
        dirty = [v for s in doc.skill_lines for v in s.values()
                 if v.startswith((":", "{", "\\")) or "}" in v or not v.strip()]
        check(f"{name}: skill values are clean", not dirty, str(dirty[:5]))

        # every editable bullet is attached to a dated entry
        orphans = [b.bid for b in editable if not b.entry_key]
        check(f"{name}: bullets attached to entries", not orphans, str(orphans))

        # a targeted edit changes only that bullet
        if editable:
            target = editable[0]
            out = render(doc, {target.bid: "Replacement bullet text for the test."}, {})
            only_one = (out != src
                        and "Replacement bullet text" in out
                        and out.count("\\item") == src.count("\\item")
                        and out[:src.index("\\begin{document}")] ==
                            src[:src.index("\\begin{document}")])
            check(f"{name}: single-bullet edit leaves everything else alone", only_one)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL TEMPLATE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
