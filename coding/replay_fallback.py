"""Replay a retrospective 25% native-reread budget on saved two-choice actions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def gap(row):
    left, right = row["arms"]["bf16"]["branch_logits"]
    return abs(left - right)


def gain(row):
    return int(row["arms"]["native"]["passed"]) - int(row["arms"]["bf16"]["passed"])


def replay(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["route"]].append(row)
    direct = sum(row["arms"]["native"]["passed"] for row in rows)
    handoff = sum(row["arms"]["bf16"]["passed"] for row in rows)
    routes = {}
    for route, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: (gap(row), row["id"]))
        if len(ordered) != 24:
            raise ValueError(f"Expected 24 actions on {route}")
        budget = len(ordered) // 4
        selected = ordered[:budget]
        threshold = gap(selected[-1])
        below = [row for row in ordered if gap(row) < threshold]
        tied = [row for row in ordered if gap(row) == threshold]
        slots = budget - len(below)
        fixed_gain = sum(gain(row) for row in below)
        tie_gains = sorted(gain(row) for row in tied)
        routes[route] = {
            "selected": [row["id"] for row in selected],
            "wrong_to_pass": sum(gain(row) == 1 for row in selected),
            "pass_to_wrong": sum(gain(row) == -1 for row in selected),
            "min_gain_over_ties": fixed_gain + sum(tie_gains[:slots]),
            "max_gain_over_ties": fixed_gain + sum(tie_gains[-slots:]),
        }
    fixes = sum(r["wrong_to_pass"] for r in routes.values())
    costs = sum(r["pass_to_wrong"] for r in routes.values())
    return {
        "decisions": len(rows), "fallback_count": sum(len(r["selected"]) for r in routes.values()),
        "handoff_passes": handoff, "native_passes": direct, "fallback_passes": handoff + fixes - costs,
        "wrong_to_pass": fixes, "pass_to_wrong": costs,
        "fallback_pass_range_over_boundary_ties": [
            handoff + sum(r[key] for r in routes.values()) for key in ("min_gain_over_ties", "max_gain_over_ties")
        ],
        "expected_passes_uniform_random_fallback": handoff + (direct - handoff) / 4,
        "routes": routes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="?", default=Path(__file__).with_name("toolmode-results.json"))
    args = parser.parse_args()
    by_panel = defaultdict(list)
    for row in json.loads(args.scores.read_bytes())["ledger"]:
        by_panel[row["panel"]].append(row)
    print(json.dumps({name: replay(rows) for name, rows in sorted(by_panel.items())}, indent=2))
