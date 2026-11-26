"""Score a two-candidate constrained decoder against the executable coding fixtures."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tarfile

from score_coding import ARMS, HERE, PANELS, ROUTES, digest, fixture_oracle, paired_interval, parse_action, validate_manifest


def records(path):
    if path.is_dir():
        for kind in ("pairs", "toolmode"):
            for file in sorted((path/kind).glob("*/*.json")):
                yield kind, file.parent.name, file.stem, file.read_bytes()
    else:
        with tarfile.open(path) as archive:
            for member in archive:
                file = Path(member.name)
                if (member.isfile() and file.suffix==".json" and len(file.parts)>=3
                        and file.parts[-3] in ("pairs", "toolmode")):
                    yield file.parts[-3], file.parent.name, file.stem, archive.extractfile(member).read()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer_ids(value):
    return isinstance(value,list) and bool(value) and all(type(i) is int and i>=0 for i in value)


def candidates_for(case, raw):
    candidates = raw["candidates"]
    require(isinstance(candidates, list) and len(candidates)==2, "Expected exactly two candidates")
    for candidate, symbol in zip(candidates, case["choices"]):
        text = json.dumps({"symbol":symbol},separators=(",",":"))
        require(candidate["symbol"]==symbol and candidate["text"]==text
                and parse_action(text,case["choices"])==(symbol,"valid") and integer_ids(candidate["ids"]),
                f"Bad candidate surface or ID sequence for {case['id']}")
    left, right = [c["ids"] for c in candidates]
    first_difference = next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),None)
    require(first_difference is not None, "One candidate is a prefix of the other or the tokenizations are identical")
    require(raw["common_ids"]==left[:first_difference] and raw["branch_index"]==first_difference
            and raw["branch_ids"]==[left[first_difference],right[first_difference]],
            "Recorded branch does not match the longest common token prefix")
    return candidates


def score_arm(case, raw, free, name, candidates, oracle):
    arm = raw["arms"][name]
    logits = arm["branch_logits"]
    require(isinstance(logits,list) and len(logits)==2 and all(
        isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in logits),
        "Expected two finite unmasked logits")
    difference = logits[0]-logits[1]
    chosen = max(range(2), key=lambda i:(logits[i], -raw["branch_ids"][i]))
    candidate = candidates[chosen]
    require(arm["choice_index"]==chosen and arm["symbol"]==candidate["symbol"] and arm["text"]==candidate["text"]
            and arm["ids"]==candidate["ids"] and arm["finish"]=="grammar_complete",
            "Chosen JSON or token sequence disagrees with greedy permitted-token argmax")
    require(math.isclose(arm["margin_first_minus_second"],difference,rel_tol=1e-6,abs_tol=1e-6)
            and arm["prompt_tokens"]==free["arms"][name]["prompt_tokens"]+len(raw["common_ids"]),
            "Recorded margin or decision-position prompt length is inconsistent")
    reference = arm["reference_argmax_id"]
    unrestricted = arm["unconstrained_argmax_id"]
    require(type(unrestricted) is int and unrestricted>=0 and (reference is None or reference==unrestricted),
            "Invalid unrestricted next-token witness")
    if unrestricted in raw["branch_ids"]:
        # An unrestricted token tied with the alternative must also be optimal among the two allowed branches.
        require(logits[raw["branch_ids"].index(unrestricted)]==max(logits),
                "Unconstrained argmax among allowed branches contradicts the branch logits")
    gold_index = case["choices"].index(case["gold"])
    behavior = oracle[candidate["symbol"]]
    passed = behavior["target_pass"] and behavior["guard_pass"]
    return {"symbol":candidate["symbol"], "text":candidate["text"], "choice_index":chosen,"passed":passed,
            "guard_regression":not behavior["guard_pass"], "behavior":behavior,"branch_logits":logits,
            "margin_gold_minus_foil":logits[gold_index]-logits[1-gold_index], "tie":logits[0]==logits[1],
            "unrestricted_argmax_was_allowed":unrestricted in raw["branch_ids"],
            "unrestricted_reference_verified":reference is not None}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"],row["route"]].append(row)
    out = {"case_directions":len(rows), "family_route_pairs":len(groups),
           "distinct_native_decisions":len({(r["id"],r["target"]) for r in rows}), "arms":{}}
    for name in ARMS:
        names = []
        for group in groups.values():
            require(len(group)==2 and group[0]["gold"]!=group[1]["gold"],"Wiring variants missing or identical")
            picks = {r["arms"][name]["symbol"] for r in group}
            successes = sum(r["arms"][name]["passed"] for r in group)
            if len(picks)==1:
                require(successes==1,"Fixed choice must pass exactly one rewired variant")
                names.append("unchanged_symbol")
            else:
                require(successes in (0,2),"Flipping a choice must pass zero or both counterbalanced variants")
                names.append("both_correct" if successes==2 else "both_reversed")
        arms = [r["arms"][name] for r in rows]
        out["arms"][name] = {"passed":sum(a["passed"] for a in arms),
            "wrong_edits":sum(not a["passed"] for a in arms),"guard_regressions":sum(a["guard_regression"] for a in arms),
            "ties":sum(a["tie"] for a in arms),
            "unrestricted_reference_checks":sum(a["unrestricted_reference_verified"] for a in arms),
            "unrestricted_argmax_allowed":sum(a["unrestricted_argmax_was_allowed"] for a in arms),
            "wiring_sensitivity":{k:names.count(k) for k in ("both_correct","unchanged_symbol","both_reversed")}}
    n = out["case_directions"]
    losses = sum(r["arms"]["native"]["passed"] and not r["arms"]["bf16"]["passed"] for r in rows)
    gains = sum(not r["arms"]["native"]["passed"] and r["arms"]["bf16"]["passed"] for r in rows)
    out.update(lost_passes=losses,gained_passes=gains,unchanged_symbols_between_arms=n-losses-gains,
               bf16_minus_native_pass=paired_interval([(r["family"],int(r["arms"]["bf16"]["passed"])
                                                       -int(r["arms"]["native"]["passed"])) for r in rows]))
    return out


def report(args):
    blob = args.cases.read_bytes()
    cases = json.loads(blob)
    validate_manifest(cases)
    index = {c["id"]:c for c in cases}
    oracle = {k:fixture_oracle(c) for k,c in index.items()}
    manifest_hash = digest(blob)
    tool_hash = digest((HERE/"run_toolmode.py").read_bytes())
    free_hash = digest((HERE/"run_coding.py").read_bytes())
    protocol_hash = digest((HERE/"toolmode-protocol.md").read_bytes())
    routes = ["__".join(pair) for pair in ROUTES]
    expected = {(route,case_id) for route in routes for case_id in index}
    pairs, tools = {}, {}
    for kind, route, case_id, raw in records(args.predictions):
        key = route,case_id
        dest = pairs if kind=="pairs" else tools
        require(key in expected and key not in dest,f"Unexpected or duplicate {kind} record {key}")
        dest[key] = raw
    require(set(pairs)==expected==set(tools),f"Expected {len(expected)} pairs and constrained records, got {len(pairs)}/{len(tools)}")
    ledger, native_identity, native_results, checks = [], {}, defaultdict(list), Counter()
    for route, case_id in sorted(expected):
        key = route,case_id
        raw, free = json.loads(tools[key]), json.loads(pairs[key])
        case = index[case_id]
        src,tgt = route.split("__")
        shared = {"id":case_id,"source":src,"target":tgt,"manifest_sha256":manifest_hash,
                  "document_sha256":digest(case["document"].encode()),"question_sha256":digest(case["question"].encode())}
        require(all(raw[k]==v==free[k] for k,v in shared.items()),f"Changed task/document/route for {key}")
        require(raw["first_record_sha256"]==digest(pairs[key]) and raw["base_runner_sha256"]==free_hash==free["runner_sha256"]
                and raw["protocol_sha256"]==protocol_hash and raw["runner_sha256"]==tool_hash
                and all(raw[k]==free[k] for k in ("source_revision","target_revision")),
                f"Changed first record, model revision, protocol or inference implementation for {key}")
        require(all(raw[k]==free[k] and re_hash(raw[k]) for k in (
                "target_input_sha256","target_prefix_sha256","after_ids_sha256"))
                and set(raw["arms"])==set(ARMS)==set(free["arms"]),f"Target inputs or arms changed for {key}")
        candidates = candidates_for(case,raw)
        require(all(len(c["ids"])<=free["max_tokens"] for c in candidates),"Grammar exceeds output budget")
        arms = {name:score_arm(case,raw,free,name,candidates,oracle[case_id]) for name in ARMS}
        native_key = tgt,case_id
        witness = (raw["target_input_sha256"],candidates,raw["common_ids"])
        require(native_identity.setdefault(native_key,witness)==witness,
                f"Native input differs across senders for {native_key}")
        native_results[native_key].append(raw["arms"]["native"])
        require(all(a["unrestricted_reference_verified"] == (case_id==cases[0]["id"]) for a in arms.values()),
                "Reference argmax witness differs from the scheduled first case")
        checks.update((route,name) for name,a in arms.items() if a["unrestricted_reference_verified"])
        ledger.append({"id":case_id,"panel":case["panel"],"family":case["family"],"variant":case["variant"],
            "base_id":case["base_id"],"route":route,"source":src,"target":tgt,"gold":case["gold"],"foil":case["foil"],
            "first_record_sha256":raw["first_record_sha256"],"toolmode_record_sha256":digest(tools[key]),
            "toolmode_record":f"toolmode/{route}/{case_id}.json","branch_index":raw["branch_index"],
            "branch_ids":raw["branch_ids"],"candidates":candidates,"arms":arms})
    require(all(checks[route,name]>0 for route in routes for name in ARMS),"Missing free-argmax reference for a route/arm")
    panels = {}
    for panel in PANELS:
        group = [r for r in ledger if r["panel"]==panel]
        panels[panel] = {"all":summarize(group),
            "by_route":{route:summarize([r for r in group if r["route"]==route]) for route in routes},
            "by_family":{f:summarize([r for r in group if r["family"]==f]) for f in sorted({r["family"] for r in group})}}
    matched = defaultdict(dict)
    for r in ledger:
        matched[r["family"],r["base_id"],r["route"]][r["panel"]] = int(r["arms"]["bf16"]["passed"])-int(r["arms"]["native"]["passed"])
    interaction = paired_interval([(family,v[PANELS[1]]-v[PANELS[0]]) for (family,_,_),v in matched.items()])
    repeat_choices = sum(a["choice_index"]==b["choice_index"] for a,b in native_results.values())
    repeat_logits = sum(a["branch_logits"]==b["branch_logits"] for a,b in native_results.values())
    return {"manifest_sha256":manifest_hash,"protocol_sha256":protocol_hash,
            "toolmode_runner_sha256":tool_hash,"free_runner_sha256":free_hash,
            "scorer_sha256":digest(Path(__file__).read_bytes()),"native_repeat_pairs":len(native_identity),
            "native_repeat_identical_choices":repeat_choices,"native_repeat_identical_logits":repeat_logits,
            "panels":panels,"archived_minus_current_change_in_bf16_minus_native_pass":interaction,"ledger":ledger}


def re_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions",type=Path,help="Directory or tar archive with pairs/ and toolmode/")
    parser.add_argument("--cases",type=Path,default=HERE/"coding-panels.json")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(args),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
