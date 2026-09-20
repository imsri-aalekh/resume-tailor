"""
Offline end-to-end test of the tailoring pipeline.

Uses StubLLM so the whole loop — write, validate, critique, revise, score —
runs without an API key. The stub deliberately misbehaves in round 1 (invents a
metric, claims Kubernetes, stuffs keywords) so we can prove the guards catch it
and the second round is cleaner.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import tailor
from core.ats import score as ats_score
from core.diffing import to_html, word_diff
from core.jd_extract import heuristic_analysis
from core.latexdoc import parse, render, verify_template_integrity
from core.llm import StubLLM
from core.matcher import build_report
from core.render import compile_tex, shim_class, available_engines

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------
# stub responses
# --------------------------------------------------------------------------

ROUND1 = {
    "edits": [
        {"id": "b001",
         "new_text": "Designed and operate Java 21 microservices on the managed-solutions and financial-accounts teams of a wealth management platform serving retail investors.",
         "changed": True, "keywords_placed": ["microservices", "Java"],
         "rationale": "Surfaces Java and microservices, both must-haves.",
         "truth_basis": "Original already says microservices on these teams."},
        # deliberately bad: invents a metric that appears nowhere
        {"id": "b002",
         "new_text": "Migrated REST APIs from AWS Lambda to ECS, cutting p99 latency by 63\\% for account operations.",
         "changed": True, "keywords_placed": ["REST APIs", "AWS"],
         "rationale": "Adds a number.", "truth_basis": "none"},
        # deliberately bad: claims Kubernetes, which is not in the vocabulary
        {"id": "b004",
         "new_text": "Built a Kubernetes-native cluster-wide circuit breaker for Orion and Schwab money-movement API calls, preventing cascading failures.",
         "changed": True, "keywords_placed": ["Kubernetes"],
         "rationale": "JD wants Kubernetes.", "truth_basis": "none"},
        {"id": "b006",
         "new_text": "Orchestrated event-driven onboarding and submit-trade flows with AWS Step Functions, replacing brittle sequential processing.",
         "changed": True, "keywords_placed": ["event-driven"],
         "rationale": "Surfaces event-driven architecture, a preferred qualification.",
         "truth_basis": "Step Functions orchestration is already in the bullet."},
    ],
    "skills_line_edits": [
        {"id": "s005", "new_values": "OOPs, Apache Kafka, System Design, Event-Driven Architecture",
         "added": ["Event-Driven Architecture"]}
    ],
    "untouched_reason": "The Amdocs bullets are less relevant to a backend infra role.",
}

CRITIQUE1 = {
    "overall_score": 61,
    "verdict": "Promising but two bullets would not survive a deep dive.",
    "bullet_verdicts": [
        {"id": "b002", "verdict": "revise", "issue": "63% latency figure is unsupported",
         "fix": "Drop the number, keep the migration and the reason", "severity": "critical"},
        {"id": "b004", "verdict": "revert", "issue": "Kubernetes is not on the resume",
         "fix": "Restore the original bullet", "severity": "critical"},
    ],
    "fabrication_flags": [
        {"id": "b002", "claim": "cutting p99 latency by 63%", "why": "Original says stabilised p99, no figure"},
        {"id": "b004", "claim": "Kubernetes-native", "why": "No Kubernetes anywhere in the resume"},
    ],
    "missed_opportunities": [
        {"requirement": "Apache Kafka", "where": "b006", "suggestion": "Name Kafka if it backed the workflows"}
    ],
    "global_notes": ["Quantification is thin outside the Freecharge role."],
}

ROUND2 = {
    "edits": [
        {"id": "b002",
         "new_text": "Migrated REST APIs from AWS Lambda to ECS, removing cold-start latency and stabilising p99 response times for account operations.",
         "changed": True, "keywords_placed": ["REST APIs", "AWS"],
         "rationale": "Removed the unsupported figure, kept the real outcome.",
         "truth_basis": "Original bullet states cold-start and p99 stabilisation."},
        {"id": "b004",
         "new_text": "Built a cluster-wide circuit breaker for Orion and Schwab money-movement API calls, preventing cascading failures during vendor outages.",
         "changed": True, "keywords_placed": ["circuit breaker"],
         "rationale": "Reverted to the original per critic.", "truth_basis": "Original."},
    ],
    "skills_line_edits": [],
}

CRITIQUE2 = {
    "overall_score": 84,
    "verdict": "Clean. Every claim traces to the original resume.",
    "bullet_verdicts": [{"id": "b001", "verdict": "keep", "issue": "", "fix": "", "severity": "minor"}],
    "fabrication_flags": [],
    "missed_opportunities": [],
    "global_notes": ["Ready to send."],
}

STUB = {
    "writer:r1": json.dumps(ROUND1),
    "critic:r1": json.dumps(CRITIQUE1),
    "reviser:r1": json.dumps(ROUND2),
    "critic:r2": json.dumps(CRITIQUE2),
}


def main():
    tex = open(os.path.join(ROOT, "samples", "aalekh_resume.tex")).read()
    jd_text = open(os.path.join(ROOT, "samples", "jd_google_backend.txt")).read()

    print("\n== parsing ==")
    doc = parse(tex)
    check("round-trips byte-identical", render(doc, {}, {}) == tex)
    check("sections found", len(doc.sections) == 4, f"{[s.norm for s in doc.sections]}")
    check("bullets found", len(doc.bullets) == 18, f"{len(doc.bullets)}")
    check("skill rows found", len(doc.skill_lines) == 5, f"{len(doc.skill_lines)}")
    check("entries linked to bullets",
          all(b.entry_key for b in doc.bullets if b.section == "experience"))

    print("\n== jd analysis ==")
    jd = heuristic_analysis(jd_text, "Senior Software Engineer, Backend Infrastructure", "Google")
    check("seniority detected", jd.seniority == "senior", jd.seniority)
    check("kafka extracted", any("kafka" in r.canonical for r in jd.requirements))
    check("requirements non-trivial", len(jd.requirements) >= 15, str(len(jd.requirements)))

    print("\n== gap report ==")
    report = build_report(doc, jd, ["Kafka", "Event Driven"])
    check("kafka not a true gap", all(g.canonical != "kafka" for g in report.true_gaps()))
    check("java flagged as listed-only",
          any(g.canonical == "java" and g.level == "listed_only" for g in report.gaps))
    check("coverage in range", 0.0 <= report.coverage()["overall"] <= 1.0,
          str(report.coverage()))

    print("\n== agent loop ==")
    client = StubLLM(handlers=STUB)
    res = tailor(doc, jd, client,
                 declared_skills=["Kafka", "Event Driven"],
                 context_notes="The EFE services run on Java 21 and Spring Boot 4.",
                 max_rounds=2)
    check("agent produced a draft", res.ok, res.error)
    check("two drafts produced", len(res.drafts) == 2, str(len(res.drafts)))

    d1 = res.drafts[0]
    print("\n  -- round 1 guards --")
    check("invented metric rejected",
          any(v.kind == "invented_metric" and v.bullet_id == "b002" for v in d1.validation.violations))
    check("unbacked Kubernetes rejected",
          any(v.kind == "unbacked_skill" and v.bullet_id == "b004" for v in d1.validation.violations))
    check("bad bullets reverted to original",
          doc.bullet("b004").text in d1.tex and "Kubernetes-native" not in d1.tex)
    check("good bullets kept", "b001" in d1.bullet_edits and "b006" in d1.bullet_edits,
          f"accepted={sorted(d1.bullet_edits)}")
    check("context-supplied version number allowed", "Java 21" in d1.tex)
    check("declared skill accepted into skills row",
          "Event-Driven Architecture" in d1.skill_edits.get("s005", ""))

    d2 = res.drafts[1]
    print("\n  -- round 2 --")
    check("round 2 clean", not d2.validation.violations, d2.validation.summary())
    check("round 2 wins", res.best is d2,
          f"composites {[d.composite for d in res.drafts]}")
    check("fabricated 63% figure never reaches the output", "63\\%" not in d2.tex)

    print("\n== template integrity ==")
    best = res.best
    check("no integrity problems", not best.integrity, str(best.integrity))
    problems = verify_template_integrity(doc, best.tex, best.bullet_edits, best.skill_edits)
    check("preamble untouched", not problems, str(problems))
    check("preamble byte-identical",
          best.tex[:best.tex.index("\\begin{document}")] == tex[:tex.index("\\begin{document}")])
    check("same number of \\item", best.tex.count("\\item") == tex.count("\\item"))
    check("documentclass unchanged",
          "\\documentclass[a4paper,11pt]{resume-openfont}" in best.tex)

    print("\n== scoring ==")
    check("ats improved", best.ats.total > res.ats_before.total,
          f"{res.ats_before.total} -> {best.ats.total}")
    check("coverage improved",
          best.report_after.coverage()["overall"] >= res.report_before.coverage()["overall"],
          f"{res.report_before.coverage()['overall']} -> {best.report_after.coverage()['overall']}")

    print("\n== diffing ==")
    ch = next(c for c in best.changes if c.bid == "b001")
    chunks = word_diff(ch.original, ch.tailored)
    check("diff has inserts", any(c.kind == "insert" for c in chunks))
    check("diff html renders", "<span" in to_html(ch.original, ch.tailored))

    print("\n== pdf ==")
    # The suite is documented as running with no LaTeX engine installed. The
    # .tex is the deliverable; the PDF is a convenience that needs pdflatex.
    # Skip rather than fail, or the whole suite goes red on a clean machine.
    if not available_engines():
        print("  [SKIP] tailored tex compiles — no LaTeX engine on this machine")
    else:
        r = compile_tex(best.tex, support_files={"resume-openfont.cls": shim_class("resume-openfont").encode()})
        check("tailored tex compiles", r.ok,
              r.friendly_error() if not r.ok else f"{len(r.pdf)} bytes")

    print("\n" + "=" * 62)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    print(f"\nATS {res.ats_before.total} -> {best.ats.total} "
          f"({res.ats_before.band()} -> {best.ats.band()})")
    for line in res.log:
        print("  ·", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
