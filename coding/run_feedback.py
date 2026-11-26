"""Continue archived coding conversations after the actual fixture tool returns a failure.

First-turn answers come from run_coding.py. A passing first edit ends that arm.
The tool discards a failed edit and permits exactly one more action. The same
original repo prefix, question IDs, and exact previous assistant IDs precede
the feedback. Only the authored fixtures run on either attempt.
"""

import argparse
import dataclasses
import json
from pathlib import Path

from cases import execute
from run_coding import HERE, ids_hash, json_to, memory_file, public_cases, sha, text_hash
from score_coding import parse_action


MODELS = ("qwen3-14b", "ministral3-14b")
PANEL = "archived_then_current"


def dispatch(case, answer):
    symbol, status = parse_action(answer["text"], case["choices"])
    observation = execute(case, symbol)
    return {"dispatch": status, "symbol": symbol, "behavior": observation,
            "passed": status == "valid" and observation["target_pass"] and observation["guard_pass"]}


def tool_feedback(case, edit):
    if edit["passed"]:
        raise ValueError("Successful edits must terminate immediately")
    if edit["dispatch"] != "valid":
        message = (f"Edit tool result: no action was dispatched ({edit['dispatch']}). "
                   "The tool requires one JSON object with a single string key named symbol "
                   "whose value is an existing integer constant in settings.py.")
    else:
        lines = ["Edit tool result: applied the change, then ran the repository tests."]
        for key in ("target", "guard"):
            obs = edit["behavior"]
            if not obs[key + "_pass"]:
                spec = case["probes"][key]
                args = ", ".join(repr(a) for a in spec["args"])
                lines.append(f"FAIL {spec['function']}({args}): expected {obs[key+'_expected']!r}, "
                             f"got {obs[key+'_after']!r}.")
        message = "\n".join(lines)
    return (message + "\nThe original working tree is restored. Read the original request and current "
            "repository. Try the edit once more. Reply with only the JSON arguments for the tool.")


def load_cases(args):
    blob = args.cases.read_bytes()
    public_cases(blob)
    rows = [r for r in json.loads(blob) if r["panel"] == PANEL]
    if len(rows) != 24 or len({r["family"] for r in rows}) != 12:
        raise ValueError("Expected the frozen set of 24 archived/current cases in 12 families")
    return sha(blob), rows


def load_first(case, args, registry, manifest_hash):
    pair = args.source + "__" + args.key
    blob = (args.out / "pairs" / pair / (case["id"] + ".json")).read_bytes()
    raw = json.loads(blob)
    expected = {"id": case["id"], "source": args.source, "target": args.key,
                "manifest_sha256": manifest_hash, "runner_sha256": sha((HERE / "run_coding.py").read_bytes()),
                "document_sha256": text_hash(case["document"]), "question_sha256": text_hash(case["question"]),
                "source_revision": registry[args.source]["revision"], "target_revision": registry[args.key]["revision"]}
    if not all(raw.get(k) == val for k, val in expected.items()) or set(raw["arms"]) != {"native", "bf16"}:
        raise ValueError(f"First-turn inputs or settings disagree for {case['id']}")
    return sha(blob), raw


def generate_retries(args, case, raw, arms, enc, torch):
    from speculative_encoding import prompts
    from speculative_encoding.encode import EncodedMemory

    memory = EncodedMemory.load(memory_file(args.out, args.source, case["document"]))
    if (memory.document != case["document"].strip() or memory.source != args.source
            or memory.fingerprint != raw["donor_fingerprint"] or memory.memory.dtype != torch.bfloat16):
        raise ValueError(f"Incorrect original memory for {case['id']}")
    prefix = enc.encode(memory=memory).full
    first = enc.question(case["question"], row_id=case["id"], metric="free", max_tokens=raw["max_tokens"])
    if (ids_hash(prefix.prefix_ids) != raw["target_prefix_sha256"]
            or (prefix.meta.get("native_prefix_ids") or prefix.prefix_ids) != prefix.prefix_ids
            or ids_hash(first.after_ids) != raw["after_ids_sha256"]
            or ids_hash(prefix.prefix_ids + first.after_ids) != raw["target_input_sha256"]):
        raise ValueError(f"First turn input tokens changed for {case['id']}")
    for arm, flow in arms.items():
        if not flow["retry_eligible"]:
            continue
        prior_ids = flow["first_answer"]["ids"]
        message = tool_feedback(case, flow["first_edit"])
        following = prompts.followup_ids(enc.target_tokenizer, message, after=prior_ids, role=memory.role)
        # Re-ingest the exact previous response token IDs, including its EOS when present.
        # The official chat helper deduplicates an end-of-turn token at the message boundary.
        suffix = list(first.after_ids) + list(prior_ids) + list(following)
        second = dataclasses.replace(first, text=message, after_ids=suffix)
        answer = enc.answer_hf(prefix, [second], native=arm == "native", top_logprobs=0)[0]
        if answer["prompt_tokens"] != len(prefix.prefix_ids) + len(suffix):
            raise ValueError(f"Retry input length mismatch for {case['id']} {arm}")
        generated = {"text": answer["text"], "ids": answer["ids"], "finish": answer["finish_reason"],
                     "prompt_tokens": answer["prompt_tokens"]}
        second_edit = dispatch(case, generated)
        flow.update(second_answer=generated, second_edit=second_edit, final_edit=second_edit,
                    retry_feedback=message, retry_turn_ids=following, first_question_ids=list(first.after_ids),
                    retry_prompt_sha256=ids_hash(prefix.prefix_ids + suffix))
        flow["transcript"].extend([{"role": "user", "content": message},
                                   {"role": "assistant", "content": generated["text"]}])


def run(args):
    from speculative_encoding import models
    from speculative_encoding.api import SpeculativeEncoder
    import torch

    registry = json.loads(args.registry.read_bytes())
    manifest_hash, rows = load_cases(args)
    signatures = {name: sha((HERE / name).read_bytes()) for name in
                  ("run_feedback.py", "run_coding.py", "cases.py", "score_coding.py")}
    first_records = [load_first(case, args, registry, manifest_hash) for case in rows]
    for key in (args.source, args.key):
        models.CATALOG[key] = dataclasses.replace(models.CATALOG[key],
            hf_id=registry[key]["repo"], revision=registry[key]["revision"])
    torch.set_num_threads(4)
    enc = None
    with torch.inference_mode():
        for i, (case, (first_sha, raw)) in enumerate(zip(rows, first_records), 1):
            path = args.out / "feedback" / (args.source + "__" + args.key) / (case["id"] + ".json")
            metadata = {"id": case["id"], "family": case["family"], "variant": case["variant"], "panel": PANEL,
                        "source": args.source, "target": args.key, "manifest_sha256": manifest_hash,
                        "first_record_sha256": first_sha, "code_sha256": signatures,
                        "document_sha256": raw["document_sha256"], "target_input_sha256": raw["target_input_sha256"],
                        "max_tokens_per_action": raw["max_tokens"]}
            if path.exists():
                previous = json.loads(path.read_bytes())
                if (any(previous.get(k) != v for k, v in metadata.items())
                        or set(previous["arms"]) != {"native", "bf16"}):
                    raise ValueError(f"Different follow-up protocol or first-generation inputs for {path}")
                continue
            arms = {}
            for arm, answer in raw["arms"].items():
                edit = dispatch(case, answer)
                arms[arm] = {"first_answer": answer, "first_edit": edit, "retry_eligible": not edit["passed"],
                             "second_answer": None, "second_edit": None, "final_edit": edit,
                             "transcript": [{"role": "user", "content": case["question"]},
                                            {"role": "assistant", "content": answer["text"]}]}
            if any(flow["retry_eligible"] for flow in arms.values()):
                if enc is None:
                    pair = models.load_model(args.key, device="cuda", check=False)
                    enc = SpeculativeEncoder.from_pretrained(args.weights, args.source, args.key, level=0,
                        load_source=False, dtype=torch.bfloat16, models={args.key: pair})
                    enc.translator.set_compiled_attention(False)
                generate_retries(args, case, raw, arms, enc, torch)
            json_to(path, {**metadata, "repository_system_message": case["document"].strip(), "arms": arms})
            print("feedback", args.source, args.key, i, "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("target",))
    parser.add_argument("key", choices=MODELS)
    parser.add_argument("--source", required=True, choices=MODELS)
    parser.add_argument("--cases", type=Path, default=HERE / "coding-panels.json")
    parser.add_argument("--registry", type=Path, default=HERE / "models.json")
    parser.add_argument("--out", type=Path, default=Path("coding-work"))
    parser.add_argument("--weights", default="trained/translator")
    args = parser.parse_args()
    if args.key == args.source:
        parser.error("Select different source and receiver models")
    run(args)
