"""Apply a saved model action to real fixture files in a fresh temporary directory."""
import argparse
from difflib import unified_diff
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from cases import make_cases, patched_settings
from score_coding import parse_action


PROBE_PROGRAM = """import app, json
from pathlib import Path
specs = json.loads(Path("probes.json").read_text())
print(json.dumps({k: getattr(app, v["function"])(*v["args"]) for k, v in specs.items()}))
"""


def materialize(case, symbol):
    with TemporaryDirectory(prefix="coding-fixture-") as folder:
        work = Path(folder)
        for name, text in case["files"].items():
            (work / name).write_text(text)
        (work / "probes.json").write_text(json.dumps(case["probes"]))
        (work / "run_checks.py").write_text(PROBE_PROGRAM)

        def run_probes():
            raw = subprocess.check_output([sys.executable, "-B", "run_checks.py"], cwd=work, text=True)
            return json.loads(raw)

        before = run_probes()
        original = (work / "settings.py").read_text()
        patch = patched_settings(case, symbol)
        (work / "settings.py").write_text(patch)
        after = run_probes()
        return {"diff": "".join(unified_diff(original.splitlines(True), patch.splitlines(True),
                                             fromfile="a/settings.py", tofile="b/settings.py")),
                "before": before, "after": after,
                "requested_test_pass": after["target"] == case["probes"]["target"]["expected"],
                "preservation_test_pass": after["guard"] == before["guard"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--case", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--arm", required=True, choices=("native", "bf16"))
    p.add_argument("--mode", choices=("free", "structured"), default="free")
    args = p.parse_args()
    route = args.source + "__" + args.target
    namespace = "pairs" if args.mode == "free" else "toolmode"
    with tarfile.open(args.archive) as archive:
        matches = [json.load(archive.extractfile(entry)) for entry in archive
                   if entry.isfile() and Path(entry.name).parts[-3:] == (namespace, route, args.case + ".json")]
    if len(matches) != 1:
        p.error("Expected exactly one first-turn prediction for that case and direction")
    fixture_id = args.case.removesuffix("--archive")
    fixture = next(case for case in make_cases() if case["id"] == fixture_id)
    generated_text = matches[0]["arms"][args.arm]["text"]
    chosen, status = parse_action(generated_text, fixture["choices"])
    result = materialize(fixture, chosen)
    print(json.dumps({"case": args.case, "route": route, "receiver_mode": args.arm, "tool_mode": args.mode,
                      "request": fixture["request"], "generated": generated_text,
                      "dispatch": status, **result}, indent=2))
