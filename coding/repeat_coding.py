"""Repeat one prospectively specified category of observed behavior per directed route.

Case selection is outcome-dependent. This checks reproducibility of six identified
observations; it does not provide a fresh estimate of error prevalence. Each route
gets an isolated directory and separate donor and receiver processes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from run_coding import HERE, json_to, memory_file, sha
from score_coding import ROUTES, fixture_oracle, inference_results, score_answer, validate_manifest


PRIORITIES = ("native_pass_bf16_valid_wrong", "native_pass_bf16_no_dispatch",
              "different_pass_status", "both_pass", "same_unsuccessful_action")


def priority(left, right):
    if left["passed"] and right["outcome"] == "wrong_edit":
        return 0
    if left["passed"] and right["outcome"] == "no_edit":
        return 1
    if left["passed"] != right["passed"]:
        return 2
    if left["passed"]:
        return 3
    if (left["symbol"], left["dispatch"]) == (right["symbol"], right["dispatch"]):
        return 4
    return None


def select(cases, records):
    by_id = {case["id"]: case for case in cases}
    oracles = {key: fixture_oracle(case) for key, case in by_id.items()}
    selected = []
    for source, target in ROUTES:
        candidates = []
        for key, case in by_id.items():
            raw = records[source, target, key]
            judged = {arm: score_answer(output, case, oracles[key]) for arm, output in raw["arms"].items()}
            rank = priority(judged["native"], judged["bf16"])
            if rank is not None:
                candidates.append((rank, key))
        if not candidates:
            raise ValueError(f"No eligible repeat control on {source} -> {target}")
        rank, key = min(candidates)
        selected.append({"source": source, "target": target, "id": key,
                         "panel": by_id[key]["panel"], "family": by_id[key]["family"],
                         "selection_reason": PRIORITIES[rank]})
    return selected


def memory_hashes(path):
    # One BF16 tensor; hash its actual bytes separately from JSON header ordering.
    with path.open("rb") as source:
        header_size = int.from_bytes(source.read(8), "little")
        if not 0 < header_size <= path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header {path}")
        header = json.loads(source.read(header_size))
        data_size = path.stat().st_size - 8 - header_size
        if (set(header) != {"__metadata__", "memory"} or header["memory"]["dtype"] != "BF16"
                or header["memory"]["data_offsets"] != [0, data_size]):
            raise ValueError(f"Expected single contiguous BF16 memory file {path}")
        hasher = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    header["__metadata__"]["info"] = json.loads(header["__metadata__"]["info"])
    canonical_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return {"tensor_sha256": hasher.hexdigest(), "metadata_sha256": sha(canonical_header), "tensor_bytes": data_size}


def compare(choice, original, fresh, mem_before, mem_after):
    fields = ("source", "target", "id", "document_sha256", "question_sha256", "target_input_sha256",
              "target_prefix_sha256", "after_ids_sha256", "runner_sha256", "source_revision",
              "target_revision", "max_tokens", "donor_fingerprint", "donor_tokens", "target_prefix_tokens")
    matches = {k: original[k] == fresh[k] for k in fields}
    arms = {}
    exact = True
    for arm in ("native", "bf16"):
        before, after = original["arms"][arm], fresh["arms"][arm]
        equal = {k: before[k] == after[k] for k in ("text", "ids", "finish", "prompt_tokens")}
        exact &= all(equal.values())
        arms[arm] = {"original": before, "repeat": after, "equal": equal}
    return {**choice, "arms": arms, "inputs_equal": matches, "all_inputs_equal": all(matches.values()),
            "both_outputs_reproduced": exact, "original_memory": mem_before, "repeat_memory": mem_after,
            "memory_equal": mem_before == mem_after, "original_target_input_sha256": original["target_input_sha256"],
            "repeat_target_input_sha256": fresh["target_input_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=HERE / "coding-panels.json")
    p.add_argument("--registry", type=Path, default=HERE / "models.json")
    p.add_argument("--out", type=Path, default=Path("coding-work"))
    p.add_argument("--repeat-work", type=Path, default=Path("coding-repeat"))
    p.add_argument("--weights", default="trained/translator")
    args = p.parse_args()
    work = args.repeat_work.resolve()
    original_dir = args.out.resolve()
    if work == original_dir or original_dir in work.parents or work in original_dir.parents:
        p.error("Choose disjoint directories for the primary and repeat runs")
    manifest = args.cases.read_bytes()
    cases = json.loads(manifest)
    validate_manifest(cases)
    runner = HERE / "run_coding.py"
    records, _, _ = inference_results(SimpleNamespace(predictions=args.out, runner=runner), cases, sha(manifest))
    selected = select(cases, records)
    by_id = {case["id"]: case for case in cases}
    for row in selected:
        source, target, key = row["source"], row["target"], row["id"]
        before = original_dir / "pairs" / (source + "__" + target) / (key + ".json")
        row["original_record_sha256"] = sha(before.read_bytes())
    selection = {"priorities": PRIORITIES, "tie_break": "ascending case ID", "items": selected,
                 "repeat_code_sha256": sha(Path(__file__).read_bytes()), "manifest_sha256": sha(manifest)}
    selection_path = work / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_bytes()) != {**selection, "priorities": list(PRIORITIES)}:
            raise ValueError("Changed selection or inputs in repeat directory")
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError(f"Fresh repeat directory required: {work}")
        json_to(selection_path, selection)
    report = []
    for choice in selected:
        s, t, key = choice["source"], choice["target"], choice["id"]
        route = s + "__" + t
        directory = work / route
        case = by_id[key]
        # The reduced manifest contains only the model-visible fields from the original.
        single = [{k: case[k] for k in ("id", "document", "question")}]
        single_file = directory / "case.json"
        json_to(single_file, single)
        original = records[s, t, key]
        flags = ["--cases", str(single_file), "--registry", str(args.registry.resolve()), "--out", str(directory),
                 "--max-tokens", str(original["max_tokens"]), "--weights", args.weights]
        print("repeat", route, key, choice["selection_reason"], flush=True)
        subprocess.run([sys.executable, str(runner), "source", s, *flags], check=True)
        subprocess.run([sys.executable, str(runner), "target", t, "--source", s, *flags], check=True)
        fresh_path = directory / "pairs" / route / (key + ".json")
        fresh_blob = fresh_path.read_bytes()
        fresh = json.loads(fresh_blob)
        if fresh["manifest_sha256"] != sha(single_file.read_bytes()):
            raise ValueError("Repeated result does not match the restricted manifest")
        mem_before = memory_hashes(memory_file(original_dir, s, case["document"]))
        mem_after = memory_hashes(memory_file(directory, s, case["document"]))
        result = compare(choice, original, fresh, mem_before, mem_after)
        result["repeat_record_sha256"] = sha(fresh_blob)
        json_to(directory / "comparison.json", result)
        report.append(result)
    summary = {"directed_routes": len(report),
               "routes_reproducing_both_generations": sum(r["both_outputs_reproduced"] for r in report),
               "routes_with_identical_inputs": sum(r["all_inputs_equal"] for r in report),
               "routes_with_identical_memory_tensor_and_metadata": sum(r["memory_equal"] for r in report)}
    json_to(work / "repeat-results.json", {"selection": selection, "summary": summary, "items": report})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
