"""Verify the selected repeat generations from archived JSON and recorded memory digests."""

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
from types import SimpleNamespace

from repeat_coding import PRIORITIES, compare, select
from run_coding import HERE, sha
from score_coding import inference_results, validate_manifest


def read_members(archive_path, requested):
    """Read named records in memory from the archive root. No tar extraction is needed."""
    found = {}
    with tarfile.open(archive_path) as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name in requested:
                if name in found or not member.isfile():
                    raise ValueError(f"Duplicate or non-file archive entry: {name}")
                found[name] = archive.extractfile(member).read()
    if set(found) != requested:
        raise ValueError(f"Missing archive entries: {sorted(requested - set(found))}")
    return found


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(args):
    manifest = args.cases.read_bytes()
    cases = json.loads(manifest)
    validate_manifest(cases)
    records, _, _ = inference_results(SimpleNamespace(predictions=args.predictions, runner=HERE/"run_coding.py"),
                                      cases, sha(manifest))
    by_id = {row["id"]:row for row in cases}
    choices = select(cases, records)
    paths = []
    required = {"selection.json"}
    for item in choices:
        route = item["source"] + "__" + item["target"]
        original = f"pairs/{route}/{item['id']}.json"
        repeated = f"{route}/{original}"
        single = route + "/case.json"
        comparison = route + "/comparison.json"
        required.update((repeated, single, comparison))
        paths.append((original, repeated, single, comparison))
    originals = read_members(args.predictions, {row[0] for row in paths})
    repeats = read_members(args.repeats, required)
    for choice, (path, _, _, _) in zip(choices, paths):
        choice["original_record_sha256"] = sha(originals[path])
    selection = {"priorities":list(PRIORITIES), "tie_break":"ascending case ID", "items":choices,
                 "repeat_code_sha256":sha((HERE/"repeat_coding.py").read_bytes()), "manifest_sha256":sha(manifest)}
    published = json.loads(args.results.read_bytes())
    check(json.loads(repeats["selection.json"]) == selection == published["selection"],
          "Selection, category ordering, checkpoint inputs or file hashes disagree")
    check(len(choices) == len(published["items"]), "Unexpected repeat report count")
    checked = []
    for choice, paths_for_item, saved in zip(choices, paths, published["items"]):
        original, repeated, single, comparison = paths_for_item
        expected_case = [{k:by_id[choice["id"]][k] for k in ("id", "document", "question")}]
        check(json.loads(repeats[single]) == expected_case, f"Repeat input changed: {single}")
        fresh = json.loads(repeats[repeated])
        check(fresh["manifest_sha256"] == sha(repeats[single]), f"Bad restricted-manifest hash: {repeated}")
        check(saved == json.loads(repeats[comparison]), f"Embedded comparison disagrees with report: {comparison}")
        # Memory files stay on the original worker. Their recorded hashes cannot verify the tensor bytes here.
        result = compare(choice, json.loads(originals[original]), fresh, saved["original_memory"], saved["repeat_memory"])
        result["repeat_record_sha256"] = sha(repeats[repeated])
        check(result == saved, f"Output or metadata comparison disagrees with independent replay: {repeated}")
        checked.append(result)
    summary = {"directed_routes":len(checked),
               "routes_reproducing_both_generations":sum(r["both_outputs_reproduced"] for r in checked),
               "routes_with_identical_inputs":sum(r["all_inputs_equal"] for r in checked),
               "routes_with_identical_memory_tensor_and_metadata":sum(r["memory_equal"] for r in checked)}
    check(summary == published["summary"], "Published summary disagrees with verified comparison records")
    return {"summary":summary, "original_repeat_arm_comparisons":2*len(checked),
            "original_and_repeat_records_rehashed":2*len(checked), "donor_tensors_rehashed_offline":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="?", default=HERE/"predictions.tar.gz",
                        help="Original archive rooted at pairs/")
    parser.add_argument("repeats", type=Path, nargs="?", default=HERE/"repeat-predictions.tar.gz",
                        help="Repeat archive rooted at selection.json and six route directories")
    parser.add_argument("--results", type=Path, default=HERE/"repeat-results.json")
    parser.add_argument("--cases", type=Path, default=HERE/"coding-panels.json")
    print(json.dumps(verify(parser.parse_args()), indent=2))
