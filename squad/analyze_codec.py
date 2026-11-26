"""Re-score saved answers and pair them by question within article clusters."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import string


def tokens(s):
    return re.sub(
        r"\b(the|an|a)\b", " ", "".join(c for c in s.lower() if c not in string.punctuation)
    ).split()


def score(text, golds):
    x = tokens(text)
    scores = []
    for gold in golds:
        y = tokens(gold)
        f1 = 2 * sum((Counter(x) & Counter(y)).values()) / (len(x) + len(y)) if x or y else 1.0
        scores.append((f1, int(x == y)))
    return max(f for f, _ in scores), max(e for _, e in scores)


def cluster_ci(values, clusters):
    grouped = defaultdict(list)
    for v, c in zip(values, clusters):
        grouped[c].append(v)
    means = [(sum(a), len(a)) for a in grouped.values()]
    rng = random.Random(9311)
    boots = []
    for _ in range(10000):
        sample = rng.choices(means, k=len(means))
        boots.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    boots.sort()
    return [round(100 * x, 2) for x in (mean(values), boots[249], boots[9749])]


def summarize(file):
    rows = [json.loads(s) for s in file.read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 64 and len({r["title"] for r in rows}) == 48
    result = {}
    for k in ("bf16", "int8", "int4", "center4", "native", "wrong"):
        recs = [r["arms"][k] for r in rows]
        refs = [r["arms"]["bf16"] for r in rows]
        scored = [score(a["text"], r["golds"]) for r, a in zip(rows, recs)]
        diff = [f - score(a["text"], r["golds"])[0] for (f, _), a, r in zip(scored, refs, rows)]
        result[k] = dict(
            f1=round(100 * mean(s[0] for s in scored), 2),
            exact=sum(s[1] for s in scored),
            same_ids=sum(a["ids"] == b["ids"] for a, b in zip(recs, refs)),
            same_first=sum(a["ids"][0] == b["ids"][0] for a, b in zip(recs, refs)),
            delta=cluster_ci(diff, [r["title"] for r in rows]),
            wins_ties_losses=[
                sum(v > 0 for v in diff),
                sum(v == 0 for v in diff),
                sum(v < 0 for v in diff),
            ],
            median_first_kl=median(a["first_kl"] for a in recs),
            total_bytes=sum(a["bytes"] for a in recs),
        )
        if "pack_ms" in recs[0]:
            result[k].update(
                median_pack_ms=median(a["pack_ms"] for a in recs),
                median_unpack_ms=median(a["unpack_ms"] for a in recs),
                median_rel_error=median(a["relative_error"] for a in recs),
            )
    result["source_tokens"] = [
        min(r["tokens"] for r in rows),
        median(r["tokens"] for r in rows),
        max(r["tokens"] for r in rows),
    ]
    return result


if __name__ == "__main__":
    files = sorted(Path(".").glob("answers-*.jsonl"))
    summary = {f.stem: summarize(f) for f in files}
    for f in sorted(Path(".").glob("storage-*.json")):
        rows = json.loads(f.read_bytes())
        assert len({r["id"] for r in rows}) == 64
        summary[f.stem] = {k: sum(r["bytes"][k] for r in rows) for k in rows[0]["bytes"]}
    Path("summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
