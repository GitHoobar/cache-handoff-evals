"""Original miniature repositories for the single-tool coding handoff study.

Run with Python 3.9+ to regenerate case-manifest.json and validate all controls.
The only generated tool argument is a constant name. Repository Python in this
file is authored fixture code; the executor accepts no code from a model.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from textwrap import dedent
from types import SimpleNamespace


@dataclass(frozen=True)
class Fixture:
    name: str
    guide: str
    symbols: tuple
    starting_values: tuple  # (target, guard) for each of two repository versions
    replacement: int
    request: str
    app: str
    function: str
    target_args: list
    expected: object
    guard_args: list


FIXTURES = (
    Fixture(
        "client-deadlines",
        "The sync client calls upload to send a cached artifact and poll to check its status. "
        "Both paths use the scheduler's integer deadline in seconds.",
        ("EDGE_BUDGET", "CORE_BUDGET"), ((5, 6), (7, 4)), 30,
        "Give artifact uploads a deadline of 30 seconds. Status polling must keep its existing deadline.",
        """
        ROUTES = {"upload": "push", "poll": "peek"}
        LIMITS = {"peek": settings.@guard@, "push": settings.@target@}

        def within_deadline(operation, elapsed_seconds):
            kind = ROUTES[operation]
            return elapsed_seconds <= LIMITS[kind]
        """,
        "within_deadline", ["upload", 20], True, ["poll", 20],
    ),
    Fixture(
        "batch-writer",
        "There are two sinks. The catalogue writer handles index events. The ledger writer handles "
        "accounting events. Each sink groups consecutive rows before handing a batch to storage.",
        ("INLINE_WIDTH", "BUFFER_WIDTH"), ((2, 3), (3, 2)), 4,
        "Group index events four per batch. Preserve accounting batch boundaries.",
        """
        SINK = {"index": "catalogue", "accounting": "ledger"}

        def batch_rows(event, rows):
            widths = {"ledger": settings.@guard@, "catalogue": settings.@target@}
            n = widths[SINK[event]]
            return [rows[i:i+n] for i in range(0, len(rows), n)]
        """,
        "batch_rows", ["index", list("abcdefg")], [list("abcd"), list("efg")],
        ["accounting", list("abcdefg")],
    ),
    Fixture(
        "history-window",
        "The activity feed has a public timeline and a personal timeline. A reader supplies "
        "the age of each entry in whole days. Returned entries preserve input order.",
        ("HOT_WINDOW", "WARM_WINDOW"), ((5, 10), (9, 7)), 30,
        "Keep public timeline entries for 30 days. Preserve the personal timeline window.",
        """
        FEEDS = {"community": "public", "mine": "personal"}
        AGE_DAYS = {"public": settings.@target@, "personal": settings.@guard@}

        def visible_ages(feed, ages):
            horizon = AGE_DAYS[FEEDS[feed]]
            return [age for age in ages if age <= horizon]
        """,
        "visible_ages", ["community", [2, 24, 30, 36]], [2, 24, 30],
        ["mine", [2, 24, 30, 36]],
    ),
    Fixture(
        "search-pages",
        "Search pages start at page one. The tablet UI uses the shelf surface. "
        "The desktop UI uses the grid surface.",
        ("COMPACT_COUNT", "FULL_COUNT"), ((2, 3), (3, 1)), 4,
        "Use four hits per page on tablets. Desktop pagination should keep its current behavior.",
        """
        SURFACE = {"tablet": "shelf", "desktop": "grid"}
        SIZE = {"grid": settings.@guard@, "shelf": settings.@target@}

        def page_hits(device, number, hits):
            size = SIZE[SURFACE[device]]
            start = (number - 1) * size
            return hits[start:start + size]
        """,
        "page_hits", ["tablet", 2, list("abcdefghij")], list("efgh"),
        ["desktop", 2, list("abcdefghij")],
    ),
    Fixture(
        "sync-attempts",
        "The sync worker records one boolean receipt for each attempted delivery. "
        "A redraw submits preview output; a publish submits final output.",
        ("RAPID_CAP", "STEADY_CAP"), ((1, 2), (3, 2)), 4,
        "Allow final output four delivery attempts. Preserve the preview attempt budget.",
        """
        def acknowledged(event, receipts):
            channel = {"redraw": "preview", "publish": "final"}[event]
            budgets = {"final": settings.@target@, "preview": settings.@guard@}
            for index, received in enumerate(receipts):
                if index >= budgets[channel]:
                    break
                if received:
                    return True
            return False
        """,
        "acknowledged", ["publish", [False, False, False, True]], True,
        ["redraw", [False, False, False, True]],
    ),
    Fixture(
        "editor-events",
        "Editor events include caret motion from the keyboard and scrubbing from the timeline. "
        "Events closer than a stream's minimum gap are coalesced into the last emitted event.",
        ("LEAD_GAP", "TRAIL_GAP"), ((2, 3), (4, 1)), 10,
        "Space emitted timeline scrub events at least 10 ms apart. Keep caret motion unchanged.",
        """
        EVENT_KIND = {"arrow": "caret", "drag": "scrub"}
        GAP = {"caret": settings.@guard@, "scrub": settings.@target@}

        def emit_times(event, times):
            out = []
            gap = GAP[EVENT_KIND[event]]
            for now in times:
                if not out or now - out[-1] >= gap:
                    out.append(now)
            return out
        """,
        "emit_times", ["drag", [0, 7, 11]], [0, 11], ["arrow", [0, 7, 11]],
    ),
    Fixture(
        "chart-smoothing",
        "Two plots share the numerical renderer. The ribbon shows measured samples and "
        "the overlay shows simulated samples. The final displayed point is the mean of a trailing window.",
        ("SERIES_SPAN", "TRACE_SPAN"), ((1, 4), (2, 5)), 3,
        "Average the last three measured samples for the ribbon. Preserve the simulated overlay.",
        """
        STREAM = {"ribbon": "probe", "overlay": "model"}
        WINDOWS = {"probe": settings.@target@, "model": settings.@guard@}

        def final_point(plot, samples):
            window = samples[-WINDOWS[STREAM[plot]]:]
            return sum(window) / len(window)
        """,
        "final_point", ["ribbon", [0, 0, 0, 0, 12, 24]], 12.0,
        ["overlay", [0, 0, 0, 0, 12, 24]],
    ),
    Fixture(
        "label-layout",
        "The asset renderer produces proof sheets and catalogue cards. A sheet places the label "
        "inside a padded frame; a card uses the same rendering helper with its own inset.",
        ("INNER_INSET", "OUTER_INSET"), ((2, 3), (1, 4)), 8,
        "Put eight pixels of padding on each side for proof sheets. Keep catalogue card geometry unchanged.",
        """
        def frame_geometry(output, content_width):
            preset = {"proof": "sheet", "catalogue": "card"}[output]
            inset = settings.@guard@ if preset == "card" else settings.@target@
            return [inset, content_width + 2 * inset]
        """,
        "frame_geometry", ["proof", 20], [8, 36], ["catalogue", 20],
    ),
    Fixture(
        "decimal-export",
        "The laboratory exporter formats the reading shown on screen separately from the "
        "reading sent with a data download. The CLI command dump produces the download.",
        ("TABLE_DIGITS", "METER_DIGITS"), ((1, 2), (4, 1)), 3,
        "Print exactly three decimal places in data downloads. Preserve the on-screen reading.",
        """
        VIEW = {"dump": "exchange", "show": "monitor"}
        DIGITS = {"monitor": settings.@guard@, "exchange": settings.@target@}

        def display_number(command, value):
            places = DIGITS[VIEW[command]]
            return format(value, "." + str(places) + "f")
        """,
        "display_number", ["dump", 12.34567], "12.346", ["show", 12.34567],
    ),
    Fixture(
        "import-preamble",
        "Two input adapters feed the importer. Spreadsheets exported by the catalogue team arrive "
        "as a sheet. Snapshots produced by the counter arrive as a tape. Each can prepend metadata rows.",
        ("FORM_OFFSET", "LOG_OFFSET"), ((1, 2), (4, 1)), 3,
        "Skip three metadata rows for catalogue spreadsheets. Keep counter snapshot imports unchanged.",
        """
        ADAPTER = {"sheet": "catalogue", "tape": "counter"}
        SKIPS = {"catalogue": settings.@target@, "counter": settings.@guard@}

        def first_data_row(source, rows):
            skip = SKIPS[ADAPTER[source]]
            return rows[skip]
        """,
        "first_data_row", ["sheet", list("abcde")], "d", ["tape", list("abcde")],
    ),
    Fixture(
        "dependency-walk",
        "The explorer builds a small tree for each build request. A release request uses the emit "
        "entry point; the interactive inspector uses the peek entry point. The root is depth zero.",
        ("NARROW_DEPTH", "BROAD_DEPTH"), ((1, 2), (2, 1)), 3,
        "For release requests include dependencies through depth three. Preserve the inspector's traversal.",
        """
        LIMITS = {"emit": settings.@target@, "peek": settings.@guard@}
        NEXT = {"app": ["codec"], "codec": ["table"], "table": ["bits"], "bits": ["atom"]}

        def visit(command, root):
            frontier = [root]
            result = list(frontier)
            for step in range(LIMITS[command]):
                frontier = [child for node in frontier for child in NEXT.get(node, [])]
                result.extend(frontier)
            return result
        """,
        "visit", ["emit", "app"], ["app", "codec", "table", "bits"], ["peek", "app"],
    ),
    Fixture(
        "trace-buffer",
        "A review session keeps the most recent actions so the UI can show history. A recording "
        "session feeds the macro editor through the same bounded-buffer helper.",
        ("FRONT_CAPACITY", "BACK_CAPACITY"), ((2, 3), (1, 2)), 4,
        "Keep the four most recent actions in review-session history. Preserve macro recordings.",
        """
        MODE = {"review": "history", "record": "macro"}

        def append_action(session, existing, new):
            limits = {"macro": settings.@guard@, "history": settings.@target@}
            combined = existing + [new]
            return combined[-limits[MODE[session]]:]
        """,
        "append_action", ["review", list("abcd"), "e"], list("bcde"),
        ["record", list("abcd"), "e"],
    ),
)

FORMAT = ('Reply with just one JSON object with the single key "symbol" and the chosen name '
          'as its string value. Do not include markdown or explanation.')


def make_cases():
    rows = []
    for index, fixture in enumerate(FIXTURES):
        for lane, variant in enumerate(("a", "b")):
            target_index = (index + lane) % 2
            target = fixture.symbols[target_index]
            guard = fixture.symbols[1 - target_index]
            target_value, guard_value = fixture.starting_values[lane]
            initial = {target: target_value, guard: guard_value}
            settings = "".join(f"{name} = {initial[name]}\n" for name in fixture.symbols)
            app = "import settings\n\n" + dedent(fixture.app).strip().replace(
                "@target@", target).replace("@guard@", guard) + "\n"
            files = {"README.md": fixture.guide + "\n", "settings.py": settings, "app.py": app}
            document = "\n\n".join(f"--- {path} ---\n{body.rstrip()}" for path, body in files.items())
            row = {
                "id": f"{fixture.name}-{variant}", "family": fixture.name, "variant": variant,
                "counterpart_id": f"{fixture.name}-{'b' if lane == 0 else 'a'}",
                "document": document, "request": fixture.request,
                "question": fixture.request + f" The edit tool sets one existing integer constant in settings.py "
                            f"to {fixture.replacement}. Choose the name it must change. " + FORMAT,
                "choices": list(fixture.symbols), "gold": target, "golds": [target], "foil": guard,
                "replacement": fixture.replacement, "files": files,
                "probes": {
                    "target": {"function": fixture.function, "args": fixture.target_args, "expected": fixture.expected},
                    "guard": {"function": fixture.function, "args": fixture.guard_args},
                },
            }
            rows.append(row)
    return rows


def patched_settings(row, symbol):
    original = row["files"]["settings.py"]
    if symbol is None:
        return original
    if symbol not in row["choices"]:
        raise ValueError(f"Unknown settings.py symbol: {symbol!r}")
    replaced, count = re.subn(rf"^{re.escape(symbol)} = [0-9]+$", f"{symbol} = {row['replacement']}",
                              original, flags=re.MULTILINE)
    if count != 1 or original == replaced:
        raise ValueError(f"Expected exactly one effective edit to {symbol}")
    return replaced


def observations(row, settings_src):
    # Fixtures are authored and pinned. The model's output is never treated as code.
    values = {}
    for line in settings_src.splitlines():
        match = re.fullmatch(r"([A-Z][A-Z_0-9]*) = ([0-9]+)", line)
        if not match or match[1] in values:
            raise ValueError(f"Invalid fixture assignment: {line}")
        values[match[1]] = int(match[2])
    if set(values) != set(row["choices"]):
        raise ValueError("Editable symbols differ from settings.py")
    module = SimpleNamespace(**values)

    def import_settings(name, globals=None, locals=None, fromlist=(), level=0):
        if name != "settings" or fromlist or level:
            raise ImportError(name)
        return module

    allowed = {"__import__": import_settings, "range": range, "len": len, "sum": sum, "list": list,
               "enumerate": enumerate, "format": format, "str": str}
    ns = {"__builtins__": allowed}
    exec(compile(row["files"]["app.py"], "app.py", "exec"), ns)
    return {key: ns[spec["function"]](*spec["args"]) for key, spec in row["probes"].items()}


def execute(row, symbol):
    before = observations(row, row["files"]["settings.py"])
    after = observations(row, patched_settings(row, symbol))
    expected = row["probes"]["target"]["expected"]
    return {"target_pass": after["target"] == expected, "guard_pass": after["guard"] == before["guard"],
            "target_before": before["target"], "target_after": after["target"], "target_expected": expected,
            "guard_before": before["guard"], "guard_after": after["guard"], "guard_expected": before["guard"]}


def validate(rows):
    indexed = {row["id"]: row for row in rows}
    assert len(indexed) == len(rows) == 24
    for row in rows:
        other = indexed[row["counterpart_id"]]
        assert other["counterpart_id"] == row["id"] and row["question"] == other["question"]
        assert other["gold"] == row["foil"] and row["gold"] != row["foil"]
        assert other["choices"] == row["choices"] and other["probes"] == row["probes"]
        for name in row["choices"]:
            assert name not in row["question"] and row["files"]["app.py"].count("settings." + name) == 1
        for symbol, target, guard in ((None, False, True), (row["gold"], True, True), (row["foil"], False, False)):
            result = execute(row, symbol)
            if (result["target_pass"], result["guard_pass"]) != (target, guard):
                raise ValueError((row["id"], symbol, result))
    for lane in ("a", "b"):
        assert sum(r["variant"] == lane and r["gold"] == r["choices"][0] for r in rows) == 6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("case-manifest.json"))
    args = parser.parse_args()
    cases = make_cases()
    validate(cases)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
