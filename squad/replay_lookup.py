"""Replay a fixed read-only country lookup from one archived SQuAD item's generations.

Run: python3 replay_lookup.py all-models/matrix-results.tar.gz --output lookup-results.json
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


ITEM = "572649d8f1498d1400e8db36"
QUESTION = "Which country is not badly hit by the embargo?"
MODELS = {"qwen3-14b", "qwen35-9b", "llama31-8b", "gemma3-12b", "gemma4-12b",
          "ministral3-8b", "ministral3-14b"}
COUNTRY_IDS = {"uk": "GB", "united kingdom": "GB", "norway": "NO",
               "sweden": "SE", "switzerland": "CH"}
REGISTRY = {key: {"id": key, "country": name} for key, name in
            (("GB", "United Kingdom"), ("NO", "Norway"), ("SE", "Sweden"), ("CH", "Switzerland"))}
ANSWER = re.compile(r"(?:the )?(uk|united kingdom|norway|sweden|switzerland)"
                    r"(?: is not badly hit by the embargo)?\.?", re.IGNORECASE)


def country_id(text):
    # Full-string matching rejects unknown answers and multiple-country responses.
    match = ANSWER.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Expected exactly one country in an accepted answer form: {text!r}")
    return COUNTRY_IDS[match.group(1).casefold()]


def replay(archive_path):
    answers = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if len(path.parts) != 4 or path.parts[0] != "matrix-results" or path.name != ITEM + ".json":
                continue
            _, kind, model, _ = path.parts
            if kind not in {"native", "pairs"}:
                continue
            if not member.isfile():
                raise ValueError(f"Expected a stored answer at {path}")
            row = json.load(archive.extractfile(member))
            if kind == "native":
                source, target = model, model
                answer = row
            else:
                source, target = row["source"], row["target"]
                if (model != f"{source}__{target}" or source == target or row["id"] != ITEM
                        or row["question"] != QUESTION or row["title"] != "1973_oil_crisis"
                        or {country_id(gold) for gold in row["golds"]} != {"GB"}):
                    raise ValueError(f"Unexpected route, item or gold at {path}")
                answer = row["arms"]["bf16"]
            if answer["finish"] != "stop" or (source, target) in answers:
                raise ValueError(f"Incomplete or duplicate answer at {path}")
            answers[source, target] = answer["text"]
    if set(answers) != set(itertools.product(MODELS, repeat=2)):
        raise ValueError("Expected all seven native answers and 42 distinct cross-model answers")

    calls = []
    summary = {}
    for (source, target), text in sorted(answers.items()):
        mode = "native" if source == target else "bf16_handoff"
        requested = country_id(text)
        record = REGISTRY[requested]
        calls.append({"mode": mode, "source": None if mode == "native" else source, "target": target,
                      "saved_answer": text, "lookup_argument": requested, "returned_record": record,
                      "correct_record": record["id"] == "GB"})
    for mode in ("native", "bf16_handoff"):
        group = [call for call in calls if call["mode"] == mode]
        correct = sum(call["correct_record"] for call in group)
        counts = Counter(call["returned_record"]["id"] for call in group)
        summary[mode] = {"calls": len(group), "records_found": sum(counts.values()),
                         "correct_records": correct, "wrong_records": len(group) - correct,
                         "returned_record_counts": dict(sorted(counts.items()))}
    return {
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "item": ITEM, "question": QUESTION, "expected_record_id": "GB",
        "scope": [
            "A retrospective replay of one selected SQuAD item across seven frozen models.",
            "Inputs are archived free-text QA generations from greedy native prefill and released BF16 translation.",
            "The script converts each saved answer into one argument using a fixed four-country vocabulary "
            "and a full-answer parser. It rejects ambiguous or unsupported text.",
            "Each parsed argument makes one local read against a fixed four-record dictionary. "
            "The gold is used only for validation and scoring.",
            "No model generated a tool call. The replay runs offline without a model, router, external tool or writes to a service.",
            "The 42 ordered handoffs share the same passage, question and model set. "
            "They do not estimate production error frequency or user impact.",
        ],
        "registry": REGISTRY, "summary": summary, "calls": calls,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="matrix-results.tar.gz from the failure-study bundle")
    parser.add_argument("--output", type=Path, help="Write JSON to this path (default: standard output)")
    args = parser.parse_args()
    result = json.dumps(replay(args.archive), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
