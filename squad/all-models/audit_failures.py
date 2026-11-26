"""Reproduce the post-hoc answer audit from saved generations. Uses Python stdlib only."""

from collections import Counter, defaultdict
import gzip
import itertools
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from analyze_codec import score


def by_question():
    native = defaultdict(dict)
    pairs = defaultdict(dict)
    with tarfile.open(ROOT / "matrix-results.tar.gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if len(parts) != 4 or not member.isfile() or not member.name.endswith(".json"):
                continue
            _, kind, model, filename = parts
            if kind == "native":
                native[Path(filename).stem][model] = json.load(archive.extractfile(member))["text"]
            elif kind == "pairs":
                row = json.load(archive.extractfile(member))
                pairs[row["id"]][row["source"], row["target"]] = row
    with gzip.open(ROOT / "stitch" / "rows.json.gz", "rt") as data:
        passages = {r["id"]: r for r in json.load(data) if r["kind"] == "natural"}
    return native, pairs, passages


def audit():
    native, pairs, passages = by_question()
    notes = json.loads((ROOT / "failure-labels.json").read_text())
    models = set(json.loads((ROOT / "models.json").read_text()))
    directions = set(itertools.permutations(models, 2))
    assert len(models) == 7 and len(pairs) == 64 and set(native) == set(pairs)
    transitions = Counter()
    labels = Counter()
    checked = {}
    unanimous_ids = []
    clear_cases = []
    observed_texts = defaultdict(set)
    for item, group in sorted(pairs.items()):
        assert set(native[item]) == models and set(group) == directions
        question = passages[item]
        golds = question["golds"]
        before = {model: score(answer, golds)[1] for model, answer in native[item].items()}
        unanimous = all(before.values())
        if unanimous:
            unanimous_ids.append(item)
        missed = defaultdict(list)
        for (source, target), row in sorted(group.items()):
            assert row["golds"] == golds and row["question"] == question["question"]
            translated = row["arms"]["bf16"]
            answer = translated["text"]
            after = score(answer, golds)[1]
            transitions[f"native_{before[target]}_translated_{after}"] += 1
            if not unanimous or after:
                continue
            # Labels apply only to the exact strings inspected in the archived passages.
            label = notes[item][answer]
            observed_texts[item].add(answer)
            labels[label] += 1
            missed[answer].append([source, target])
            if label == "wrong entity":
                phrase = answer.strip().rstrip(".").casefold()
                clear_cases.append({
                    "id": item, "source": source, "target": target, "answer": answer,
                    "stopped": translated["finish"] == "stop",
                    "verbatim_span_ignoring_case_and_final_period": phrase in question["document"].casefold(),
                    "identical_ids": {arm: row["arms"][arm]["ids"] == translated["ids"]
                                      for arm in ("int8", "int4", "shared2")},
                })
        if missed:
            checked[item] = {
                "question": question["question"], "passage": question["document"], "golds": golds,
                "native_answers": native[item],
                "losses": {text: {"label": notes[item][text], "directions": routes}
                           for text, routes in missed.items()},
            }
    assert dict(observed_texts) == {item: set(answers) for item, answers in notes.items()}
    return {
        "article_count": len({passages[item]["title"] for item in pairs}),
        "question_count": len(pairs), "directions_per_question": len(directions),
        "exact_transitions": transitions, "unanimous_native_exact_ids": unanimous_ids,
        "unanimous_translated_cells": len(unanimous_ids) * len(directions),
        "unanimous_exact_loss_labels": labels, "clear_cases": clear_cases,
        "adjudicated_questions": checked,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False, sort_keys=True))
