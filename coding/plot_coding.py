"""Render the executable fixture and the first-action results from their saved inputs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from cases import execute, make_cases


ROOT = Path(__file__).resolve().parent
PAPER = "#FCFBF7"
INK = "#192F3C"
MUTED = "#485F6A"
TEAL = "#12635F"
RUST = "#A34229"
RULE = "#C9D2CF"
MONO = "DejaVu Sans Mono"


def canvas(height=7.8):
    fig = plt.figure(figsize=(10.4, height), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, text, size, color=INK, weight="normal", **style):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, va="center", **style)


def rule(ax, y):
    ax.plot([.06, .94], [y, y], linewidth=1.15, color=RULE)


def save(fig, output):
    fig.savefig(output, dpi=170, facecolor=PAPER)
    plt.close(fig)


def plot_example(manifest, output):
    fixture = next(row for row in json.loads(manifest.read_bytes()) if row["id"] == "client-deadlines-a")
    original = next(row for row in make_cases() if row["id"] == fixture["id"])
    if fixture != original:
        raise ValueError("Example differs from the authored fixture")
    # Explicitly validate the simplified route diagram against the stored source.
    symbols = ("EDGE_BUDGET", "CORE_BUDGET")
    if (fixture["choices"] != list(symbols) or fixture["replacement"] != 30
            or fixture["files"]["settings.py"] != "EDGE_BUDGET = 5\nCORE_BUDGET = 6\n"
            or 'ROUTES = {"upload": "push", "poll": "peek"}' not in fixture["files"]["app.py"]
            or 'LIMITS = {"peek": settings.CORE_BUDGET, "push": settings.EDGE_BUDGET}' not in fixture["files"]["app.py"]):
        raise ValueError("The illustrated routes or values have changed")
    fig, ax = canvas()
    label(ax, .06, .929, "Change the upload deadline to 30 seconds", 25.2, weight="bold")
    label(ax, .06, .855, "The same edit must preserve the status-poll deadline.", 17.5)
    rule(ax, .79)
    label(ax, .06, .735, "How the current code chooses a deadline", 17, weight="bold")
    for y, (route, internal, symbol, n) in zip(
        (.642, .558), (("upload", "push", symbols[0], 5), ("poll", "peek", symbols[1], 6))
    ):
        label(ax, .065, y, route, 19, family=MONO)
        label(ax, .225, y, "→", 23, MUTED)
        label(ax, .30, y, internal, 19, family=MONO)
        label(ax, .447, y, "→", 23, MUTED)
        label(ax, .527, y, f"{symbol} = {n}", 19, family=MONO, weight="bold")
    rule(ax, .49)
    label(ax, .06, .439, "Test the fixture at 20 seconds", 17, weight="bold")
    for x, text in ((.06, "Edit in settings.py"), (.54, "Upload"), (.76, "Status poll")):
        label(ax, x, .359, text, 15.3, MUTED)
    for y, symbol in zip((.28, .212, .144), (None, *symbols)):
        result = execute(fixture, symbol)
        edit = f"{symbol} = 30" if symbol else "Original settings"
        label(ax, .06, y, edit, 16.6, family=MONO if symbol else "DejaVu Sans")
        for x, key in ((.54, "target"), (.76, "guard")):
            within = result[key + "_after"]
            correct = result[key + "_pass"]
            label(ax, x, y, "In time" if within else "Timed out", 16.4,
                  TEAL if correct else RUST, weight="bold")
    rule(ax, .086)
    label(ax, .06, .043, "Executed fixture controls  •  One original Python repository  •  No model outputs shown",
          12.2, MUTED)
    save(fig, output)


def plot_results(path, output):
    data = json.loads(path.read_bytes())
    fig, ax = canvas(7.7)
    label(ax, .06, .929, "Did the coding edit pass both tests?", 27, weight="bold")
    label(ax, .06, .852, "First tool decision on 24 original repositories in six model directions.", 16.2)
    rule(ax, .787)
    metrics = (("passed", "Passed", TEAL), ("wrong_edits", "Wrong edit", RUST),
               ("undispatched", "No edit", MUTED))
    column_centers = (.54, .708, .876)
    for x, (_, name, color) in zip(column_centers, metrics):
        label(ax, x, .73, name, 16.5, color, weight="bold", ha="center")
    panels = (("current_only", "Current repository", .647),
              ("archived_then_current", "Archive followed by current repository", .415))
    for panel, name, title_y in panels:
        summary = data["panels"][panel]["all"]
        n = summary["case_directions"]
        if (n, summary["base_repositories"], summary["families"], summary["distinct_native_decisions"]) != (144, 24, 12, 72):
            raise ValueError(f"Unexpected coverage for {name}")
        label(ax, .06, title_y, name, 16.4, weight="bold")
        for i, (method, arm_name) in enumerate((("native", "Direct reading"), ("bf16", "Cache handoff"))):
            y = title_y - .073 - i * .066
            counts = summary["arms"][method]
            if sum(counts[k] for k, _, _ in metrics) != n or counts["guard_regressions"] != counts["wrong_edits"]:
                raise ValueError(f"Outcome counts disagree in {name} / {arm_name}")
            label(ax, .07, y, arm_name, 16.2)
            for x, (key, _, color) in zip(column_centers, metrics):
                count = counts[key]
                ax.add_patch(Rectangle((x-.065, y-.025), .13, .05, facecolor="#E8ECE9", linewidth=0))
                ax.add_patch(Rectangle((x-.065, y-.025), .13*count/n, .05, facecolor=color,
                                       alpha=.22, linewidth=0))
                label(ax, x, y, str(count), 16, weight="bold", ha="center")
    rule(ax, .22)
    label(ax, .06, .165, "Each row counts 144 routes. Direct answers repeat for the two senders per receiver.", 12.3)
    label(ax, .06, .109, "A wrong edit changes the other behavior. No edit means the tool received no valid action.", 12.3)
    label(ax, .06, .053, "Qwen 3 14B, Ministral 3 14B and Gemma 3 12B  •  Greedy decoding", 12.3, MUTED)
    save(fig, output)


def plot_structured(path, output):
    data = json.loads(path.read_bytes())
    fig, ax = canvas(7.8)
    label(ax, .06, .933, "Which code edit does the handoff select?", 25.5, weight="bold")
    label(ax, .06, .856, "Two legal JSON actions. One fixes the request and preserves the other behavior.", 15.5)
    rule(ax, .79)
    label(ax, .06, .694, "Edits passing both tests", 18.2, weight="bold")
    for i, (method, title) in enumerate((("native", "Direct reading"), ("bf16", "Cache handoff"))):
        label(ax, .065, .598-i*.075, title, 17)
    rule(ax, .457)
    label(ax, .06, .407, "Same task and receiver", 18.2, weight="bold")
    label(ax, .065, .326, "Passed directly → wrong edit after handoff", 14.2, RUST)
    label(ax, .065, .249, "Wrong edit directly → passed after handoff", 14.2, TEAL)

    for panel, header, cx in (("current_only", "Current\nrepository", .66),
                              ("archived_then_current", "Archived + current\nrepository", .852)):
        report = data["panels"][panel]
        s = report["all"]
        n = s["case_directions"]
        if (n != 144 or s["distinct_native_decisions"] != 72 or len(report["by_family"]) != 12
                or len(report["by_route"]) != 6):
            raise ValueError(f"Unexpected coverage in {panel}")
        label(ax, cx, .708, header, 15.8, ha="center", multialignment="center", weight="bold")
        for i, method in enumerate(("native", "bf16")):
            arm = s["arms"][method]
            if arm["passed"] + arm["wrong_edits"] != n or arm["wrong_edits"] != arm["guard_regressions"]:
                raise ValueError("Constrained actions did not partition into passing and regressing edits")
            y = .598 - i*.075
            ax.add_patch(Rectangle((cx-.081, y-.028), .162, .056, facecolor="#E8ECE9", linewidth=0))
            ax.add_patch(Rectangle((cx-.081, y-.028), .162*arm["passed"]/n, .056, facecolor=TEAL,
                                   alpha=.22, linewidth=0))
            label(ax, cx, y, f"{arm['passed']}/{n}", 17, ha="center", weight="bold")
        lost, gained = s["lost_passes"], s["gained_passes"]
        if (s["arms"]["native"]["passed"] - lost + gained != s["arms"]["bf16"]["passed"]
                or lost + gained + s["unchanged_symbols_between_arms"] != n):
            raise ValueError("Paired flips disagree with the marginal outcomes")
        label(ax, cx, .326, str(lost), 19, RUST, weight="bold", ha="center")
        label(ax, cx, .249, str(gained), 19, TEAL, weight="bold", ha="center")
    rule(ax, .19)
    label(ax, .06, .144, "24 Python repositories across 12 tasks, including upload deadlines and retry budgets.", 12.8)
    label(ax, .06, .092, "All six directions among Qwen 3 14B, Ministral 3 14B and Gemma 3 12B.", 12.8)
    label(ax, .06, .042, "Each panel has 144 route decisions. Direct reads repeat for both senders.", 12.1, MUTED)
    save(fig, output)


def plot_observed(archive_path, panels_path, output):
    from replay_action import materialize
    from score_coding import parse_action

    case_id = "batch-writer-a"
    route = "ministral3-14b__qwen3-14b"
    panel_bytes = panels_path.read_bytes()
    panel = next(r for r in json.loads(panel_bytes) if r["id"] == case_id)
    fixture = next(r for r in make_cases() if r["id"] == case_id)
    if panel["files"] != fixture["files"] or panel["panel"] != "current_only":
        raise ValueError("Expected the unchanged current-only repository")
    with tarfile.open(archive_path) as archive:
        matches = [json.load(archive.extractfile(entry)) for entry in archive if entry.isfile()
                   and Path(entry.name).parts[-3:] == ("pairs", route, case_id + ".json")]
    if len(matches) != 1:
        raise ValueError("Expected one matching route and case")
    raw = matches[0]
    if (raw["id"] != case_id or raw["source"] + "__" + raw["target"] != route
            or raw["manifest_sha256"] != sha256(panel_bytes).hexdigest()
            or raw["document_sha256"] != sha256(panel["document"].encode()).hexdigest()
            or raw["question_sha256"] != sha256(panel["question"].encode()).hexdigest()):
        raise ValueError("Inference input differs from the illustrated fixture")
    output_by_arm = {}
    row_symbols = {}
    for arm in ("native", "bf16"):
        symbol, status = parse_action(raw["arms"][arm]["text"], fixture["choices"])
        if status != "valid":
            raise ValueError(f"Model action did not dispatch: {arm}")
        row_symbols[arm] = symbol
        run = materialize(fixture, symbol)
        output_by_arm[arm] = run
        if (run["requested_test_pass"], run["preservation_test_pass"]) != (arm == "native", arm == "native"):
            raise ValueError(f"Previously observed test result changed: {arm}")
    native = output_by_arm["native"]
    translated = output_by_arm["bf16"]
    source = list("abcdefg")

    def batches(size):
        return [source[i:i+size] for i in range(0, len(source), size)]

    base_outputs = {"target":batches(2), "guard":batches(3)}
    if (row_symbols != {"native":"BUFFER_WIDTH","bf16":"INLINE_WIDTH"}
            or native["before"] != base_outputs or translated["before"] != base_outputs
            or native["after"] != {"target":batches(4), "guard":batches(3)}
            or translated["after"] != {"target":batches(2), "guard":batches(4)}):
        raise ValueError("Selected symbols or executed partitions disagree with the caption")

    fig, ax = canvas(7.6)
    label(ax, .06, .933, "The handoff changed the wrong batch size", 25.5, weight="bold")
    label(ax, .06, .86, "Ministral 3 14B → Qwen 3 14B   •   Same Python repository and request", 15.5)
    rule(ax, .796)
    label(ax, .06, .737, "“Group index events four per batch.", 21.5, weight="bold")
    label(ax, .06, .671, "Preserve accounting batch boundaries.”", 21.5, weight="bold")
    rule(ax, .61)
    label(ax, .06, .553, "Applied settings.py edit", 16, MUTED)
    label(ax, .51, .553, "Index batches", 16, MUTED)
    label(ax, .743, .553, "Accounting batches", 16, MUTED)
    specs = (("Original repository", "", base_outputs, .458, MUTED),
             ("Qwen reads directly", row_symbols["native"], native["after"], .337, TEAL),
             ("Qwen reads the handoff", row_symbols["bf16"], translated["after"], .192, RUST))
    for title, symbol, observations, y, accent in specs:
        label(ax, .06, y, title, 16.3, accent, weight="bold")
        if symbol:
            label(ax, .06, y-.043, f"{symbol} = 4", 13.4, family=MONO)
        for x, key in ((.51,"target"),(.743,"guard")):
            batch_output = observations[key]
            label(ax, x, y, " + ".join(str(len(batch)) for batch in batch_output),
                  16.5, accent, weight="bold")
    rule(ax, .102)
    label(ax, .06, .049, "Batch sizes for seven rows. Both valid JSON actions ran on one original Python fixture.", 12.6)
    save(fig, output)


def plot_wiring_swap(path, output):
    from replay_action import materialize
    from score_coding import parse_action

    report = json.loads(path.read_bytes())
    fixtures = {case["id"]: case for case in make_cases()}
    pair = sorted((r for r in report["ledger"] if r["family"] == "batch-writer"
                   and r["panel"] == "current_only" and r["source"] == "ministral3-14b"
                   and r["target"] == "qwen3-14b"), key=lambda r: r["variant"])
    if [r["id"] for r in pair] != ["batch-writer-a", "batch-writer-b"]:
        raise ValueError("Expected both matched current-only variants for the illustrated route")

    fig, ax = canvas(7.7)
    label(ax, .06, .93, "Which setting controls index batches?", 27, weight="bold")
    label(ax, .06, .856, "Ministral 3 14B → Qwen 3 14B   •   Two versions of one Python fixture", 15.4)
    rule(ax, .795)
    label(ax, .06, .735, "“Group index events four per batch.", 20, weight="bold")
    label(ax, .06, .673, "Preserve accounting batch boundaries.”", 20, weight="bold")
    for x, row in zip((.546, .818), pair):
        fixture = fixtures[row["id"]]
        if (fixture["request"] != "Group index events four per batch. Preserve accounting batch boundaries."
                or fixture["files"]["settings.py"] != "INLINE_WIDTH = 3\nBUFFER_WIDTH = 2\n"
                or (row["gold"], row["foil"]) != (fixture["gold"], fixture["foil"])
                or fixture["replacement"] != 4
                or f'"catalogue": settings.{row["gold"]}' not in fixture["files"]["app.py"]
                or f'"ledger": settings.{row["foil"]}' not in fixture["files"]["app.py"]):
            raise ValueError(f"Saved code or target disagrees with the illustrated wiring: {row['id']}")
        label(ax, x, .562, f"Version {row['variant'].upper()}", 17.7, weight="bold", ha="center")
        label(ax, x, .481, row["gold"], 16.2, family=MONO, ha="center")
        label(ax, x, .412, row["foil"], 16.2, family=MONO, ha="center")
        for arm, y, correct in (("native", .277, True), ("bf16", .153, False)):
            evidence = row["arms"][arm]
            symbol, status = parse_action(evidence["text"], fixture["choices"])
            run = materialize(fixture, symbol)
            if (status != "valid" or symbol != evidence["symbol"]
                    or evidence["passed"] != correct
                    or run["requested_test_pass"] != correct or run["preservation_test_pass"] != correct
                    or run["after"]["target"] != evidence["behavior"]["target_after"]
                    or run["after"]["guard"] != evidence["behavior"]["guard_after"]):
                raise ValueError(f"Saved action or its behavioral effect changed: {row['id']} / {arm}")
            color = TEAL if correct else RUST
            label(ax, x, y, symbol, 16.4, color, weight="bold", family=MONO, ha="center")
            label(ax, x, y - .044, "Both tests pass" if correct else "Both tests fail",
                  13.6, color, ha="center")
    rule(ax, .621)
    label(ax, .06, .481, "Index uses", 17)
    label(ax, .06, .412, "Accounting uses", 17)
    rule(ax, .351)
    label(ax, .06, .277, "Qwen reads directly", 16.4, weight="bold")
    label(ax, .06, .153, "Qwen after handoff", 16.4, weight="bold")
    rule(ax, .074)
    label(ax, .06, .037, "Both versions start with INLINE_WIDTH = 3 and BUFFER_WIDTH = 2. All four actions ran.", 12.1)
    save(fig, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    example = sub.add_parser("example")
    example.add_argument("--cases", type=Path, default=ROOT / "case-manifest.json")
    example.add_argument("--output", type=Path, default=ROOT / "coding-example.png")
    results = sub.add_parser("results")
    results.add_argument("--results", type=Path, default=ROOT / "coding-results.json")
    results.add_argument("--output", type=Path, default=ROOT / "coding-outcomes.png")
    structured = sub.add_parser("structured")
    structured.add_argument("--results", type=Path, default=ROOT / "toolmode-results.json")
    structured.add_argument("--output", type=Path, default=ROOT / "coding-structured.png")
    observed = sub.add_parser("observed")
    observed.add_argument("--archive", type=Path, default=ROOT / "predictions.tar.gz")
    observed.add_argument("--panels", type=Path, default=ROOT / "coding-panels.json")
    observed.add_argument("--output", type=Path, default=ROOT / "coding-batch-writer.png")
    wiring = sub.add_parser("wiring")
    wiring.add_argument("--results", type=Path, default=ROOT / "coding-results.json")
    wiring.add_argument("--output", type=Path, default=ROOT / "coding-wiring-swap.png")
    args = p.parse_args()
    if args.command == "example":
        plot_example(args.cases, args.output)
    elif args.command == "results":
        plot_results(args.results, args.output)
    elif args.command == "structured":
        plot_structured(args.results, args.output)
    elif args.command == "observed":
        plot_observed(args.archive, args.panels, args.output)
    else:
        plot_wiring_swap(args.results, args.output)
