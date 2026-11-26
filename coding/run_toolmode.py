"""Select a coding tool argument with greedy decoding restricted to two JSON strings."""

import argparse
import dataclasses
import json
import math
from pathlib import Path

from run_coding import DEFAULT_MODELS, HERE, ids_hash, json_to, memory_file, public_cases, sha, text_hash


def grammar(tokenizer, symbols, stops):
    if (len(symbols) != 2 or not all(isinstance(s, str) and s for s in symbols)
            or symbols[0] == symbols[1]):
        raise ValueError("Expected two different named tool arguments")
    candidates = []
    for symbol in symbols:
        text = json.dumps({"symbol": symbol}, ensure_ascii=False, separators=(",", ":"))
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text or not ids or any(token in stops for token in ids):
            raise ValueError(f"Could not encode the exact legal JSON: {text!r}, decoded as {decoded!r}")
        candidates.append({"symbol": symbol, "text": text, "ids": ids})
    aa, bb = (c["ids"] for c in candidates)
    for i, (left, right) in enumerate(zip(aa, bb)):
        if left != right:
            return candidates, aa[:i], [left, right]
    raise ValueError("The canonical action token sequences need a divergent token before either terminates")


def winner(logits, branch):
    if len(logits) != 2 or not all(math.isfinite(f) for f in logits):
        raise ValueError("Expected two finite branch logits")
    return max(range(2), key=lambda j: (logits[j], -branch[j]))


def constrained(enc, prefix, question, common, branch, candidates, arm, check_reference, torch):
    from speculative_encoding.encode import prefix_to
    from speculative_encoding.serving import hf_engine

    target = enc.target
    translated = None if arm == "native" else prefix_to(prefix.prefix, enc.device)
    cache, prefix_length = hf_engine.build_cache(target, translated, prefix.prefix_ids)
    suffix = list(question.after_ids) + common
    next_position = prefix_length + len(suffix)
    ids = torch.tensor(suffix, dtype=torch.long, device=target.device)
    positions = torch.arange(prefix_length, next_position, device=target.device)
    last_hidden = target.forward(ids, positions, cache)[-1]
    vocab_logits = target.logits(last_hidden).float()
    if vocab_logits.ndim != 1:
        raise ValueError(f"Expected one vector of vocabulary logits, got {vocab_logits.shape}")
    branch_logits = vocab_logits[branch].tolist()
    top_id = int(vocab_logits.argmax())
    chosen = winner(branch_logits, branch)
    action = candidates[chosen]
    del cache, translated, last_hidden, vocab_logits
    reference_id = None
    if check_reference:
        probe = dataclasses.replace(question, after_ids=suffix, max_tokens=1)
        reference = enc.answer_hf(prefix, [probe], native=arm == "native", top_logprobs=0)[0]
        if len(reference["ids"]) != 1 or reference["prompt_tokens"] != next_position:
            raise ValueError("The reference next-token probe did not see the exact branch context")
        reference_id = reference["ids"][0]
        if reference_id != top_id:
            raise ValueError(f"Released unrestricted greedy step chose {reference_id}; raw argmax was {top_id}")
    return {"text": action["text"], "ids": action["ids"], "symbol": action["symbol"], "choice_index": chosen,
            "branch_logits": branch_logits, "margin_first_minus_second": branch_logits[0] - branch_logits[1],
            "unconstrained_argmax_id": top_id, "reference_argmax_id": reference_id,
            "finish": "grammar_complete", "prompt_tokens": next_position}


def run(args):
    import torch
    from speculative_encoding import models, prompts
    from speculative_encoding.api import SpeculativeEncoder
    from speculative_encoding.encode import EncodedMemory
    from speculative_encoding.serving.hf_engine import stop_ids

    registry = json.loads(args.registry.read_bytes())
    manifest = args.cases.read_bytes()
    # Only the same public inputs and the schema enum are used for model inference.
    rows = public_cases(manifest)
    symbols = {r["id"]: r["choices"] for r in json.loads(manifest)}
    manifest_hash = sha(manifest)
    base_runner_sha = sha((HERE / "run_coding.py").read_bytes())
    runner_sha = sha(Path(__file__).read_bytes())
    protocol_sha = sha((HERE / "toolmode-protocol.md").read_bytes())
    route = args.source + "__" + args.key
    for key in (args.source, args.key):
        models.CATALOG[key] = dataclasses.replace(models.CATALOG[key],
            hf_id=registry[key]["repo"], revision=registry[key]["revision"])
    torch.set_num_threads(4)
    with torch.inference_mode():
        pair = models.load_model(args.key, device="cuda", check=False)
        enc = SpeculativeEncoder.from_pretrained(args.weights, args.source, args.key, level=0,
            load_source=False, dtype=torch.bfloat16, models={args.key: pair})
        enc.translator.set_compiled_attention(False)
        tok = enc.target_tokenizer
        stops = set(stop_ids(enc.target, tok))
        for index, row in enumerate(rows):
            key = row["id"]
            first_path = args.out / "pairs" / route / (key + ".json")
            first_blob = first_path.read_bytes()
            first = json.loads(first_blob)
            originals = {"id": key, "source": args.source, "target": args.key, "manifest_sha256": manifest_hash,
                         "document_sha256": text_hash(row["document"]), "question_sha256": text_hash(row["question"]),
                         "source_revision": registry[args.source]["revision"], "target_revision": registry[args.key]["revision"]}
            if any(first.get(k) != v for k, v in originals.items()) or first["runner_sha256"] != base_runner_sha:
                raise ValueError(f"Free generation inputs have changed: {first_path}")
            identity = {**originals, "runner_sha256": runner_sha, "base_runner_sha256": base_runner_sha,
                        "protocol_sha256": protocol_sha, "first_record_sha256": sha(first_blob)}
            path = args.out / "toolmode" / route / (key + ".json")
            if path.exists():
                prior = json.loads(path.read_bytes())
                if (any(prior.get(k) != v for k, v in identity.items()) or set(prior["arms"]) != {"native", "bf16"}):
                    raise ValueError(f"Existing constrained result used different inputs: {path}")
                continue
            memory = EncodedMemory.load(memory_file(args.out, args.source, row["document"]))
            if (memory.document != row["document"].strip() or memory.source != args.source
                    or memory.fingerprint != first["donor_fingerprint"] or memory.memory.dtype != torch.bfloat16):
                raise ValueError(f"Incorrect original BF16 donor memory for {key}")
            prefix = enc.encode(memory=memory).full
            question = enc.question(row["question"], row_id=key, metric="free", max_tokens=first["max_tokens"])
            direct = prompts.render(tok, prompts.chat_row("coding", key, row["document"], question=row["question"], metric="free"))
            if (prefix.prefix_ids != direct.header_ids + direct.doc_ids or question.after_ids != direct.after_ids
                    or (prefix.meta.get("native_prefix_ids") or prefix.prefix_ids) != prefix.prefix_ids):
                raise ValueError(f"Direct and translated context mismatch for {key}")
            provenance = {"target_prefix_sha256": ids_hash(prefix.prefix_ids), "after_ids_sha256": ids_hash(question.after_ids),
                          "target_input_sha256": ids_hash(prefix.prefix_ids + question.after_ids)}
            if any(first.get(k) != v for k, v in provenance.items()):
                raise ValueError(f"Different prompt from first generation for {key}")
            candidates, common, branch = grammar(tok, symbols[key], stops)
            if any(len(c["ids"]) > first["max_tokens"] for c in candidates):
                raise ValueError("Tool grammar cannot finish within the base generation cap")
            arms = {a: constrained(enc, prefix, question, common, branch, candidates, a, index == 0, torch)
                    for a in ("native", "bf16")}
            json_to(path, {**identity, **provenance, "candidates": candidates, "common_ids": common,
                           "branch_index": len(common), "branch_ids": branch, "arms": arms})
            print("toolmode", route, index + 1, "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("target",))
    parser.add_argument("key", choices=DEFAULT_MODELS)
    parser.add_argument("--source", required=True, choices=DEFAULT_MODELS)
    parser.add_argument("--cases", type=Path, default=HERE / "coding-panels.json")
    parser.add_argument("--registry", type=Path, default=HERE / "models.json")
    parser.add_argument("--out", type=Path, default=Path("coding-work"))
    parser.add_argument("--weights", default="trained/translator")
    args = parser.parse_args()
    if args.source == args.key:
        parser.error("Use a different source and receiver")
    run(args)
