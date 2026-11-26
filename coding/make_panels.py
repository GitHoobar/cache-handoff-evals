"""Build matched current-only and archive-plus-current coding contexts."""
import argparse
import json
from pathlib import Path

from cases import validate


ROOT = Path(__file__).resolve().parent


def panels(rows):
    validate(rows)
    index = {r["id"]: r for r in rows}
    result = []
    for panel in ("current_only", "archived_then_current"):
        for row in rows:
            current = "CURRENT WORKING TREE (edit this version)\n\n" + row["document"]
            if panel == "archived_then_current":
                old = index[row["counterpart_id"]]
                document = ("ARCHIVED WORKING TREE (superseded, read-only)\n\n"
                            + old["document"] + "\n\nEND OF ARCHIVED WORKING TREE\n\n" + current)
                case_id = row["id"] + "--archive"
            else:
                document, case_id = current, row["id"]
            result.append({**row, "document": document, "id": case_id,
                           "base_id": row["id"], "panel": panel})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=ROOT / "case-manifest.json")
    p.add_argument("--output", type=Path, default=ROOT / "coding-panels.json")
    args = p.parse_args()
    args.output.write_text(json.dumps(panels(json.loads(args.cases.read_text())),
                                      indent=2, ensure_ascii=False) + "\n")
