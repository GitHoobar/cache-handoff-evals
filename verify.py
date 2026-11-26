"""Recompute the published results and execute the four batch-writer edits, offline."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
CODING = ROOT / "coding"
SQUAD = ROOT / "squad"


def run(folder, script, *args):
    return subprocess.check_output([sys.executable, "-B", script, *args], cwd=folder)


def compare_report(actual, reference):
    if actual != reference.read_bytes():
        raise ValueError(f"Recomputed output differs from {reference.relative_to(ROOT)}")
    print(f"Verified {reference.relative_to(ROOT)}")


def check_scores(work):
    for filename, script, inputs in (
        ("case-manifest.json", "cases.py", []),
        ("coding-panels.json", "make_panels.py", ["--cases", work / "case-manifest.json"]),
        ("coding-results.json", "score_coding.py", ["predictions.tar.gz"]),
        ("toolmode-results.json", "score_toolmode.py", ["predictions.tar.gz"]),
        ("feedback-results.json", "score_feedback.py", ["predictions.tar.gz"]),
        ("diagnostics.json", "diagnostics.py", [work / "coding-results.json"]),
    ):
        output = work / filename
        run(CODING, script, *inputs, "--output", output)
        compare_report(output.read_bytes(), CODING / filename)
    print(run(CODING, "verify_repeats.py").decode().strip())

    lookup = work / "lookup.json"
    run(SQUAD, "replay_lookup.py", "all-models/matrix-results.tar.gz", "--output", lookup)
    compare_report(lookup.read_bytes(), SQUAD / "lookup-results.json")
    audit = run(SQUAD, "all-models/audit_failures.py")
    compare_report(audit, SQUAD / "all-models/failure-audit.json")


def check_patches():
    for case, correct, alternative in (
        ("batch-writer-a", "BUFFER_WIDTH", "INLINE_WIDTH"),
        ("batch-writer-b", "INLINE_WIDTH", "BUFFER_WIDTH"),
    ):
        for arm, expected_symbol, should_pass in (("native", correct, True), ("bf16", alternative, False)):
            result = json.loads(run(
                CODING, "replay_action.py", "predictions.tar.gz", "--case", case,
                "--source", "ministral3-14b", "--target", "qwen3-14b", "--arm", arm,
            ))
            action = json.loads(result["generated"])
            tests = (result["requested_test_pass"], result["preservation_test_pass"])
            if action != {"symbol": expected_symbol} or tests != (should_pass, should_pass):
                raise ValueError(f"Unexpected edit or behavior on {case}, {arm}")
            print(f"{case} {arm}: {expected_symbol}, {sum(tests)}/2 behavioral tests passed")


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        check_scores(Path(temp))
    check_patches()
