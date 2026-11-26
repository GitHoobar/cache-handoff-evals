"""Verify and score the archived-context coding agent's one-retry trajectories."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import re
import tarfile

from run_coding import ids_hash
from run_feedback import MODELS, PANEL, tool_feedback
from score_coding import ARMS, HERE, digest, fixture_oracle, paired_interval, score_answer, validate_manifest


def raw_records(path):
    """Read raw bytes so a trajectory can be bound to the exact first-generation file."""
    if path.is_dir():
        for kind in ("pairs", "feedback"):
            for file in sorted((path/kind).glob("*/*.json")):
                yield kind, file.parent.name, file.stem, file.read_bytes()
    else:
        with tarfile.open(path) as archive:
            for mem in archive:
                p = Path(mem.name)
                if mem.isfile() and len(p.parts) >= 3 and p.parts[-3] in ("pairs", "feedback") and p.suffix == ".json":
                    yield p.parts[-3], p.parent.name, p.stem, archive.extractfile(mem).read()


def check_answer(answer, cap):
    if (not isinstance(answer, dict) or not isinstance(answer.get("text"), str)
            or not isinstance(answer.get("ids"), list) or any(type(v) is not int for v in answer["ids"])
            or len(answer["ids"]) > cap or not isinstance(answer.get("finish"), str)
            or type(answer.get("prompt_tokens")) is not int or answer["prompt_tokens"] <= 0):
        raise ValueError("Invalid raw answer or token count")


def verified_edit(case, answer, actual, oracle):
    scored = score_answer(answer, case, oracle)
    check = {k: scored[k] for k in ("dispatch", "symbol", "behavior", "passed")}
    if actual != check:
        raise ValueError(f"Saved edit disagrees with the independent replay for {case['id']}")
    return scored


def verify_arm(case, first, raw, arm, oracle):
    flow = raw["arms"][arm]
    cap = first["max_tokens"]
    start = first["arms"][arm]
    check_answer(start, cap)
    if flow["first_answer"] != start:
        raise ValueError("First saved answer differs between runs")
    original = verified_edit(case, start, flow["first_edit"], oracle)
    retry = not original["passed"]
    if flow["retry_eligible"] is not retry:
        raise ValueError("Incorrect retry eligibility")
    transcript = [{"role": "user", "content": case["question"]}, {"role": "assistant", "content": start["text"]}]
    final = original
    if retry:
        expected_message = tool_feedback(case, flow["first_edit"])
        if any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", expected_message) for name in case["choices"]):
            raise ValueError(f"Retry message discloses a candidate in {case['id']}")
        if expected_message != flow["retry_feedback"]:
            raise ValueError("Feedback was not the deterministic parser/test output")
        answer = flow["second_answer"]
        check_answer(answer, cap)
        q_ids, turn = flow["first_question_ids"], flow["retry_turn_ids"]
        prefix_n = first["target_prefix_tokens"]
        if (ids_hash(q_ids) != first["after_ids_sha256"] or not isinstance(turn, list) or not turn
                or any(type(i) is not int for i in q_ids + turn)
                or start["prompt_tokens"] != prefix_n + len(q_ids)
                or answer["prompt_tokens"] != start["prompt_tokens"] + len(start["ids"]) + len(turn)
                or not re.fullmatch("[0-9a-f]{64}", flow["retry_prompt_sha256"])):
            raise ValueError("Retry prompt changed the prior question or has an inconsistent token count")
        if "retry_prefix_ids" in flow:
            prefix = flow["retry_prefix_ids"]
            if (len(prefix) != prefix_n or ids_hash(prefix) != first["target_prefix_sha256"]
                    or ids_hash(prefix + q_ids) != first["target_input_sha256"]
                    or ids_hash(prefix + q_ids + start["ids"] + turn) != flow["retry_prompt_sha256"]):
                raise ValueError("Retry input digest does not match its stored token sequence")
        final = verified_edit(case, answer, flow["second_edit"], oracle)
        transcript.extend([{"role": "user", "content": expected_message},
                           {"role": "assistant", "content": answer["text"]}])
    elif (flow["second_answer"] is not None or flow["second_edit"] is not None
          or any(k in flow for k in ("retry_feedback", "retry_prompt_sha256", "retry_turn_ids"))):
        raise ValueError("Passing first action unexpectedly triggered a retry")
    if (flow["transcript"] != transcript or flow["final_edit"] != (flow["second_edit"] if retry else flow["first_edit"])):
        raise ValueError("Transcript or saved final edit disagrees with the observed trajectory")
    return {"first": original, "final": final, "attempts": 1 + int(retry), "recovered": retry and final["passed"],
            "retry_feedback": flow.get("retry_feedback"), "retry_prompt_sha256": flow.get("retry_prompt_sha256"),
            "retry_full_digest_rechecked": retry and "retry_prefix_ids" in flow}


def summarize(rows):
    result = {"case_directions":len(rows), "families":len({r["family"] for r in rows}), "arms":{}}
    for name in ARMS:
        flows = [r["arms"][name] for r in rows]
        res = {"retries":sum(f["attempts"] == 2 for f in flows), "recovered":sum(f["recovered"] for f in flows),
               "actions_generated":sum(f["attempts"] for f in flows)}
        for stage in ("first", "final"):
            arms = [f[stage] for f in flows]
            res[stage] = {
                "passed":sum(a["passed"] for a in arms),
                "guard_regressions":sum(a["guard_regression"] for a in arms),
                "wrong_edits":sum(a["outcome"] == "wrong_edit" for a in arms),
                "undispatched":sum(a["outcome"] == "no_edit" for a in arms),
                "dispatch_counts":dict(sorted(Counter(a["dispatch"] for a in arms).items())),
            }
        res["first_to_final"] = dict(sorted(Counter(f["first"]["outcome"]+"_to_"+f["final"]["outcome"]
                                                     for f in flows).items()))
        result["arms"][name] = res
    for stage in ("first", "final"):
        result[stage+"_comparison"] = {
            "native_pass_to_bf16_failure":sum(
                r["arms"]["native"][stage]["passed"] and not r["arms"]["bf16"][stage]["passed"] for r in rows),
            "native_failure_to_bf16_pass":sum(
                not r["arms"]["native"][stage]["passed"] and r["arms"]["bf16"][stage]["passed"] for r in rows),
            "bf16_minus_native_pass":paired_interval([
                (r["family"], int(r["arms"]["bf16"][stage]["passed"]) - int(r["arms"]["native"][stage]["passed"]))
                for r in rows]),
        }
    return result


def report(args):
    blob = args.cases.read_bytes()
    cases = json.loads(blob)
    validate_manifest(cases)
    index = {r["id"]:r for r in cases if r["panel"] == PANEL}
    routes = tuple("__".join(pair) for pair in itertools.permutations(MODELS, 2))
    required = set(itertools.product(routes, index))
    first, feedback = {}, {}
    for kind, route, case_id, content in raw_records(args.input):
        key = route, case_id
        if kind == "feedback" and key not in required:
            raise ValueError(f"Unexpected feedback trajectory {key}")
        if key not in required:
            continue
        records = first if kind == "pairs" else feedback
        if key in records:
            raise ValueError(f"Duplicate {kind} item {key}")
        records[key] = content
    if set(first) != required or set(feedback) != required:
        raise ValueError(f"Expected {len(required)} archived pairs and feedback trajectories, found {len(first)}/{len(feedback)}")
    signatures = {n:digest((HERE/n).read_bytes()) for n in ("run_feedback.py", "run_coding.py", "cases.py", "score_coding.py")}
    manifest_hash = digest(blob)
    checked = {key:fixture_oracle(case) for key,case in index.items()}
    ledger = []
    for route, case_id in sorted(required):
        key = route, case_id
        raw = json.loads(feedback[key])
        baseline = json.loads(first[key])
        case = index[case_id]
        source, target = route.split("__")
        shared = {"source":source, "target":target, "id":case_id, "manifest_sha256":manifest_hash,
                  "document_sha256":digest(case["document"].encode())}
        expected = {**shared,"first_record_sha256":digest(first[key]),"code_sha256":signatures,
                    "panel":PANEL,"family":case["family"],"variant":case["variant"],
                    "max_tokens_per_action":baseline["max_tokens"],"target_input_sha256":baseline["target_input_sha256"],
                    "repository_system_message":case["document"].strip()}
        if (any(raw[k] != v for k,v in expected.items()) or any(baseline[k] != v for k,v in shared.items())
                or baseline["runner_sha256"] != signatures["run_coding.py"]
                or baseline["question_sha256"] != digest(case["question"].encode())
                or set(baseline["arms"]) != set(ARMS) or set(raw["arms"]) != set(ARMS)):
            raise ValueError(f"Changed reference inputs for {key}")
        arms = {arm:verify_arm(case, baseline, raw, arm, checked[case_id]) for arm in ARMS}
        ledger.append({"id":case_id,"family":case["family"],"variant":case["variant"],"route":route,
                       "first_record":f"pairs/{route}/{case_id}.json","feedback_record":f"feedback/{route}/{case_id}.json",
                       "first_record_sha256":digest(first[key]),"feedback_record_sha256":digest(feedback[key]),"arms":arms})
    return {"manifest_sha256":manifest_hash,"code_sha256":signatures,"scorer_sha256":digest(Path(__file__).read_bytes()),
            "panel":PANEL,"all":summarize(ledger),
            "by_route":{route:summarize([r for r in ledger if r["route"]==route]) for route in routes},"ledger":ledger}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory or tar archive containing pairs/ and feedback/")
    parser.add_argument("--cases", type=Path, default=HERE/"coding-panels.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(args),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
