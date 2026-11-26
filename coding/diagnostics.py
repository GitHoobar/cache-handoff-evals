"""Inspect how a model responds when the same original issue is wired to the opposite setting."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def diagnostic(ledger, strip_fence):
    groups = defaultdict(list)
    for row in ledger:
        groups[row["source"], row["target"], row["family"]].append(row)
    out = {}
    for arm in ("native", "bf16"):
        counts = Counter()
        by_case = Counter()
        for rows in groups.values():
            assert len(rows) == 2 and {r["variant"] for r in rows} == {"a", "b"}
            a, b = sorted(rows, key=lambda r: r["variant"])
            assert a["gold"] == b["foil"] and b["gold"] == a["foil"]
            symbols = []
            for row in (a, b):
                observed = row["arms"][arm]
                sym = observed["symbol"]
                if strip_fence and sym is None:
                    sym = observed["fenced_symbol_diagnostic"]
                symbols.append(sym)
                by_case["no_action" if sym is None else "pass" if sym == row["gold"] else "wrong_edit"] += 1
            if None in symbols:
                result = "at_least_one_undispatched"
            elif symbols[0] == symbols[1]:
                result = "same_symbol_despite_changed_wiring"
            elif symbols == [a["gold"], b["gold"]]:
                result = "both_correct"
            else:
                assert symbols == [a["foil"], b["foil"]]
                result = "both_wrong"
            counts[result] += 1
        out[arm] = {"decisions": dict(sorted(by_case.items())), "counterbalanced_pairs": dict(sorted(counts.items()))}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.results.read_text())["ledger"]
    report = {}
    for panel in sorted({r["panel"] for r in rows}):
        group = [r for r in rows if r["panel"] == panel]
        report[panel] = {"strict": diagnostic(group, False),
                         "ignoring_one_json_fence": diagnostic(group, True)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
