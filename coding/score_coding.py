"""Replay generated single-argument edits against authored Python fixtures."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import tarfile


HERE = Path(__file__).resolve().parent
PANELS = ("current_only", "archived_then_current")
MODELS = ("qwen3-14b", "ministral3-14b", "gemma3-12b")
ROUTES = tuple(itertools.permutations(MODELS, 2))
ARMS = ("native", "bf16")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate key")
        obj[key] = value
    return obj


def parse_action(text, choices):
    """Only the entire, exactly schema-valid response may dispatch a tool action."""
    if text.strip().casefold() in ("", "abstain", "none", "no action"):
        return None, "abstained"
    try:
        obj = json.loads(text.strip(), object_pairs_hook=unique_object)
    except ValueError:
        return None, "malformed_json"
    if not isinstance(obj, dict) or set(obj) != {"symbol"} or not isinstance(obj["symbol"], str):
        return None, "wrong_schema"
    if obj["symbol"] not in choices:
        return None, "unknown_symbol"
    return obj["symbol"], "valid"


def fenced_diagnostic(text, choices):
    fence = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n?```", text.strip(), re.DOTALL | re.IGNORECASE)
    return parse_action(fence.group(1), choices)[0] if fence else None


def fixture_oracle(case):
    from cases import execute, make_cases

    choices = case["choices"]
    trusted = next((row for row in make_cases() if row["id"] == case.get("base_id", case["id"])), None)
    fields = ("choices", "files", "probes", "replacement", "gold", "foil", "question", "family", "variant")
    if trusted is None or any(case.get(key) != trusted[key] for key in fields):
        raise ValueError(f"Case differs from authored executable fixture: {case['id']}")
    if len(choices) != 2 or set(choices) != {case["gold"], case["foil"]} or len(set(choices)) != 2:
        raise ValueError(f"Invalid fixture choices: {case['id']}")
    expected = ((None, False, True), (case["gold"], True, True), (case["foil"], False, False))
    checked = {}
    for symbol, target_ok, guard_ok in expected:
        output = execute(trusted, symbol)
        if output["target_pass"] is not target_ok or output["guard_pass"] is not guard_ok:
            raise ValueError(f"Broken fixture oracle: {case['id']} {symbol!r} {output!r}")
        checked[symbol] = output
    return checked


def score_answer(raw, case, oracle):
    symbol, dispatch = parse_action(raw["text"], case["choices"])
    # There are just three possible fixture executions. Reuse their audited outputs.
    observed = oracle[symbol]
    successful = symbol is not None and observed["target_pass"] and observed["guard_pass"]
    outcome = "passed" if successful else "no_edit" if symbol is None else "wrong_edit"
    return {
        "text": raw["text"], "finish": raw["finish"], "dispatch": dispatch,
        "symbol": symbol, "outcome": outcome, "passed": successful,
        "guard_regression": not observed["guard_pass"], "behavior": observed,
        "fenced_symbol_diagnostic": fenced_diagnostic(raw["text"], case["choices"]),
    }


def read_pairs(path):
    if path.is_dir():
        root = path if path.name == "pairs" else path / "pairs"
        for pair in sorted(root.glob("*/*.json")):
            yield pair.parent.name, pair.stem, json.loads(pair.read_bytes())
    else:
        with tarfile.open(path) as archive:
            for member in archive:
                parts = Path(member.name).parts
                if member.isfile() and len(parts) >= 3 and parts[-3] == "pairs" and parts[-1].endswith(".json"):
                    yield parts[-2], Path(parts[-1]).stem, json.load(archive.extractfile(member))


def validate_manifest(cases):
    ids = [row["id"] for row in cases]
    if len(ids) != len(set(ids)) or {r["panel"] for r in cases} != set(PANELS):
        raise ValueError("Expected unique IDs and both frozen panels")
    panels = {p: {row["base_id"]: row for row in cases if row["panel"] == p} for p in PANELS}
    current, archived = (panels[p] for p in PANELS)
    if len(cases) != 48 or len(current) != 24 or current.keys() != archived.keys():
        raise ValueError("Expected 24 matched base repositories in each panel")
    families = defaultdict(list)
    for base_id, case in current.items():
        other = archived[base_id]
        stable_fields = ("question", "variant", "family", "choices", "files", "replacement", "probes", "gold", "foil")
        if any(case[k] != other[k] for k in stable_fields) or case["document"] == other["document"]:
            raise ValueError(f"Unmatched task or duplicate document across panels for {base_id}")
        families[case["family"]].append(case)
    if len(families) != 12 or any(len(v) != 2 or len({r["variant"] for r in v}) != 2 for v in families.values()):
        raise ValueError("Expected two distinct variants in each of 12 families")


def inference_results(args, cases, case_hash):
    available = {row["id"]: row for row in cases}
    signature = digest(args.runner.read_bytes())
    inputs, repeats, metadata, records = {}, {}, set(), {}
    for file_route, file_id, raw in read_pairs(args.predictions):
        source, target, case_id = raw["source"], raw["target"], raw["id"]
        key = (source, target, case_id)
        if (key in records or (source, target) not in ROUTES or file_route != f"{source}__{target}"
                or file_id != case_id or case_id not in available):
            raise ValueError(f"Duplicate or unexpected record {key}")
        case = available[case_id]
        if (raw["manifest_sha256"] != case_hash or raw["runner_sha256"] != signature
                or raw["document_sha256"] != digest(case["document"].encode())
                or raw["question_sha256"] != digest(case["question"].encode())):
            raise ValueError(f"Changed inputs or inference code for {key}")
        arms = raw["arms"]
        if set(arms) != set(ARMS) or arms["native"]["prompt_tokens"] != arms["bf16"]["prompt_tokens"]:
            raise ValueError(f"Missing arms or unequal input lengths for {key}")
        for arm in arms.values():
            if (not isinstance(arm["text"], str) or not isinstance(arm["ids"], list)
                    or any(type(i) is not int for i in arm["ids"])
                    or len(arm["ids"]) > raw["max_tokens"] or not isinstance(arm["finish"], str)):
                raise ValueError(f"Malformed generated answer for {key}")
        target_case = (target, case_id)
        identity = (raw["target_input_sha256"], raw["target_prefix_sha256"], raw["after_ids_sha256"],
                    arms["native"]["prompt_tokens"], raw["target_revision"])
        if inputs.setdefault(target_case, identity) != identity:
            raise ValueError(f"Native targets received different inputs across senders for {target_case}")
        if repeats.setdefault(target_case, arms["native"]) != arms["native"]:
            raise ValueError(f"Replicated native generations differ for {target_case}; inspect before pooling")
        metadata.add((source, raw["source_revision"], raw["max_tokens"]))
        metadata.add((target, raw["target_revision"], raw["max_tokens"]))
        records[key] = raw
    if len(metadata) != len(MODELS) or len({row[2] for row in metadata}) != 1:
        raise ValueError("Model revisions or output cap changed across runs")
    if set(records) != {(s, t, c) for s, t in ROUTES for c in available}:
        raise ValueError(f"Incomplete matrix: expected {len(ROUTES) * len(cases)} observations, found {len(records)}")
    return records, len(repeats), metadata


def paired_interval(values):
    """Paired difference, weighting observations equally and resampling whole families."""
    if not values:
        return None
    grouped = defaultdict(list)
    for family, value in values:
        grouped[family].append(value)
    blocks = [(sum(grouped[f]), len(grouped[f])) for f in sorted(grouped)]
    estimate = 100 * sum(total for total, n in blocks) / len(values)
    if all(v == values[0][1] for _, v in values):
        lo = hi = estimate
    else:
        rng = random.Random(74309)
        reps = []
        for _ in range(10000):
            sample = rng.choices(blocks, k=len(blocks))
            reps.append(100 * sum(total for total, n in sample) / sum(n for total, n in sample))
        reps.sort()
        lo, hi = reps[249], reps[9749]
    return {"percentage_points": round(estimate, 3), "family_bootstrap_95": [round(lo, 3), round(hi, 3)]}


def summaries(rows):
    result = {"case_directions": len(rows), "families": len({r["family"] for r in rows}),
              "base_repositories": len({r["base_id"] for r in rows}),
              "distinct_native_decisions": len({(r["id"], r["target"]) for r in rows}), "arms": {}}
    for name in ARMS:
        arms = [r["arms"][name] for r in rows]
        result["arms"][name] = {
            "passed": sum(a["passed"] for a in arms),
            "wrong_edits": sum(a["outcome"] == "wrong_edit" for a in arms),
            "guard_regressions": sum(a["guard_regression"] for a in arms),
            "undispatched": sum(a["outcome"] == "no_edit" for a in arms),
            "dispatch_counts": dict(sorted(Counter(a["dispatch"] for a in arms).items())),
            "stops": dict(sorted(Counter(a["finish"] for a in arms).items())),
            "valid_only_after_fence_removed": sum(a["fenced_symbol_diagnostic"] is not None for a in arms),
        }
    transitions = Counter((r["arms"]["native"]["outcome"], r["arms"]["bf16"]["outcome"]) for r in rows)
    result["transitions_native_to_bf16"] = {f"{a}_to_{b}": n for (a,b), n in sorted(transitions.items())}
    result["lost_passes"] = sum(n for (a,b), n in transitions.items() if a == "passed" and b != a)
    result["gained_passes"] = sum(n for (a,b), n in transitions.items() if b == "passed" and a != b)
    result["bf16_minus_native_pass"] = paired_interval([
        (r["family"], int(r["arms"]["bf16"]["passed"]) - int(r["arms"]["native"]["passed"])) for r in rows])
    return result


def report(args):
    contents = args.cases.read_bytes()
    cases = json.loads(contents)
    validate_manifest(cases)
    records, n_unique, revisions = inference_results(args, cases, digest(contents))
    rows = []
    # The oracle executes only the authored fixture and one of its finite prevalidated arguments.
    # Output strings, including fenced JSON diagnostics, never reach a Python interpreter.
    for case in sorted(cases, key=lambda r: r["id"]):
        oracle = fixture_oracle(case)
        for source, target in ROUTES:
            raw = records[source, target, case["id"]]
            arms = {name: score_answer(raw["arms"][name], case, oracle) for name in ARMS}
            rows.append({k: case[k] for k in ("id", "base_id", "family", "variant", "panel", "gold", "foil")}
                        | {"source":source, "target":target, "arms":arms,
                           "identical_answer_ids":raw["arms"]["native"]["ids"] == raw["arms"]["bf16"]["ids"]})
    passed = {(r["id"], r["target"]): r["arms"]["native"]["passed"] for r in rows}
    by_panel = {}
    for panel in PANELS:
        group = [r for r in rows if r["panel"] == panel]
        witnessed = [r for r in group if passed[r["id"],r["target"]] and passed[r["id"],r["source"]]]
        by_panel[panel] = {
            "all": summaries(group), "both_endpoints_native_passed": summaries(witnessed),
            "by_route": {f"{s}__{t}":summaries([r for r in group if (r["source"],r["target"]) == (s,t)])
                         for s,t in ROUTES},
            "by_family": {f:summaries([r for r in group if r["family"] == f])
                          for f in sorted({r["family"] for r in group})},
        }
    def differences(panel):
        return {(r["base_id"], r["source"], r["target"]): (r["family"],
                int(r["arms"]["bf16"]["passed"]) - int(r["arms"]["native"]["passed"]))
                for r in rows if r["panel"] == panel}
    curr, archive = map(differences, PANELS)
    interaction = [(family, v - curr[k][1]) for k,(family,v) in archive.items()]
    return {"manifest_sha256":digest(contents), "runner_sha256":digest(args.runner.read_bytes()),
            "scorer_sha256":digest(Path(__file__).read_bytes()), "fixture_engine_sha256":digest((HERE/"cases.py").read_bytes()),
            "unique_native_cases_across_panels":n_unique, "verified_native_repeat_pairs":n_unique,
            "model_revisions_and_cap":sorted(revisions), "panels":by_panel,
            "archived_minus_current_change_in_bf16_minus_native_pass":paired_interval(interaction), "ledger":rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", type=Path, help="Work directory, pairs directory, or archive containing pairs/")
    ap.add_argument("--cases", type=Path, default=HERE / "coding-panels.json")
    ap.add_argument("--runner", type=Path, default=HERE / "run_coding.py")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = report(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
