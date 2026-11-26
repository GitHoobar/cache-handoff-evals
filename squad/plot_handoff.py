"""Render the X figure from the saved 7 native and 42 translated model outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


QUESTION_ID = "572649d8f1498d1400e8db36"
ALIASES = {
    "uk": "UK", "the uk": "UK", "united kingdom": "UK",
    "the uk is not badly hit by the embargo": "UK",
    "norway": "Norway", "switzerland": "Switzerland", "sweden": "Sweden",
}


def load(archive):
    controls, handoffs = {}, {}
    with tarfile.open(archive) as tar:
        for entry in tar:
            path = Path(entry.name)
            if not entry.isfile() or path.name != QUESTION_ID + ".json":
                continue
            _, kind, key, _ = path.parts
            if kind not in ("native", "pairs"):
                continue
            record = json.load(tar.extractfile(entry))
            if kind == "native":
                target, answer = controls, record["text"]
            else:
                assert record["question"] == "Which country is not badly hit by the embargo?"
                target, answer = handoffs, record["arms"]["bf16"]["text"]
            assert key not in target
            target[key] = ALIASES[answer.strip().casefold().rstrip(".")]
    assert len(controls) == 7 and set(handoffs) == {
        s + "__" + t for s in controls for t in controls if s != t}
    return Counter(controls.values()), Counter(handoffs.values())


def draw(archive, output):
    native, counts = load(archive)
    trials = sum(counts.values())
    ink, muted, warm, teal = "#192F3C", "#475C67", "#A34229", "#12635F"
    fig = plt.figure(figsize=(9.2, 7.3), facecolor="#FCFBF7")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()

    def label(x, y, value, size, *, color=ink, weight="normal", **style):
        ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                va="center", **style)

    label(.065, .932, "Which country escaped the worst", 28, weight="bold")
    label(.065, .862, "of the oil embargo?", 28, weight="bold")
    label(.065, .766, f"All {native['UK']} models answered UK when they read the passage.", 16)
    ax.plot([.065, .935], [.712, .712], color="#C9D2CF", lw=1.2)
    label(.065, .647, "Answer after cache handoff", 17, weight="bold")
    label(.935, .647, "Routes", 14, ha="right", color=muted)

    for country, y in zip(("UK", "Norway", "Switzerland", "Sweden"), (.54, .441, .342, .243)):
        color = teal if country == "UK" else warm
        label(.065, y, country, 17, weight="bold", color=color)
        width = .515
        left = .325
        ax.add_patch(Rectangle((left, y-.022), width, .044, facecolor="#E6E8E2"))
        ax.add_patch(Rectangle((left, y-.022), width*counts[country]/trials, .044, facecolor=color))
        label(.935, y, str(counts[country]), 18, weight="bold", ha="right", color=color)
    ax.plot([.065, .935], [.15, .15], color="#C9D2CF", lw=1.2)
    label(.065, .097, f"One SQuAD question • 7 models • {trials} directed handoffs", 14.5)
    label(.065, .048, "Unchanged passage, released BF16 translator, greedy decoding.", 12, color=muted)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.matrix_archive, args.output)
