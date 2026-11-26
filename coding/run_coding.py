"""Generate free-text coding actions with a native or translated repository prefix.

Run one source process per donor, then one target process per direction. The
manifest can also contain oracle data; only the document and question are sent
to models. Each completed pair of generations is saved before the next case.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = ("qwen3-14b", "ministral3-14b", "gemma3-12b")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_hash(text):
    return sha(text.encode("utf-8"))


def ids_hash(ids):
    return text_hash(json.dumps(list(ids), separators=(",", ":")))


def save(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        partial = Path(temp.name)
    try:
        writer(partial)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def json_to(path, obj):
    save(path, lambda dest: dest.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n"))


def public_cases(blob):
    rows = [{key: row[key] for key in ("id", "document", "question")} for row in json.loads(blob)]
    names = [r["id"] for r in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError("Expected at least one case and unique IDs")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["id"]):
            raise ValueError(f"Unsafe item ID {row['id']!r}")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("document", "question")):
            raise ValueError(f"Empty document or question in {row['id']}")
    return rows


def memory_file(out, source, document):
    return out / "memories" / source / (text_hash(document) + ".safetensors")


def source_stage(args, rows, torch, models):
    from speculative_encoding.api import MultiTargetEncoder
    from speculative_encoding.encode import EncodedMemory

    pair = models.load_model(args.key, device="cuda", check=False)
    enc = MultiTargetEncoder.from_pretrained(args.weights, args.key, [args.key],
        level=0, dtype=torch.bfloat16, models={args.key: pair}, offload_targets=True)
    enc.translator.set_compiled_attention(False)
    documents = list(dict.fromkeys(row["document"] for row in rows))
    for index, document in enumerate(documents, 1):
        path = memory_file(args.out, args.key, document)
        if path.exists():
            memory = EncodedMemory.load(path)
        else:
            _, memory = enc.shared_input(document, share="memory")
        if memory.document != document.strip() or memory.source != args.key:
            raise ValueError(f"Source altered or truncated document {path.stem}")
        if not path.exists():
            save(path, lambda partial: memory.save(partial, dtype=torch.bfloat16))
        print("source", args.key, index, "/", len(documents), memory.tokens, flush=True)


def target_stage(args, rows, metadata, torch, models):
    from speculative_encoding import prompts
    from speculative_encoding.api import SpeculativeEncoder
    from speculative_encoding.encode import EncodedMemory

    pair = models.load_model(args.key, device="cuda", check=False)
    enc = SpeculativeEncoder.from_pretrained(args.weights, args.source, args.key, level=0,
        load_source=False, dtype=torch.bfloat16, models={args.key: pair})
    enc.translator.set_compiled_attention(False)
    dest = args.out / "pairs" / (args.source + "__" + args.key)
    for index, row in enumerate(rows, 1):
        path = dest / (row["id"] + ".json")
        inputs = {**metadata, "id": row["id"], "source": args.source, "target": args.key,
                  "document_sha256": text_hash(row["document"]), "question_sha256": text_hash(row["question"])}
        if path.exists():
            previous = json.loads(path.read_bytes())
            if not all(previous.get(k) == v for k, v in inputs.items()) or set(previous["arms"]) != {"native", "bf16"}:
                raise ValueError(f"Previous output used different settings or inputs: {path}")
            continue
        memory = EncodedMemory.load(memory_file(args.out, args.source, row["document"]))
        if memory.document != row["document"].strip() or memory.source != args.source or memory.memory.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected donor memory for {row['id']}")
        prefix = enc.encode(memory=memory).full
        q = enc.question(row["question"], row_id=row["id"], metric="free", max_tokens=args.max_tokens)
        direct = prompts.render(enc.target_tokenizer,
            prompts.chat_row("coding", row["id"], row["document"], question=row["question"], metric="free"))
        native_ids = prefix.meta.get("native_prefix_ids") or prefix.prefix_ids
        if (native_ids != prefix.prefix_ids or prefix.prefix_ids != direct.header_ids + direct.doc_ids
                or q.after_ids != direct.after_ids or q.text != row["question"]):
            raise ValueError(f"Direct and translated target input tokens disagree for {row['id']}")
        item = {**inputs, "donor_tokens": memory.tokens, "donor_fingerprint": memory.fingerprint,
                "target_prefix_tokens": len(prefix.prefix_ids), "target_prefix_sha256": ids_hash(prefix.prefix_ids),
                "after_ids_sha256": ids_hash(q.after_ids), "target_input_sha256": ids_hash(direct.native_ids),
                "arms": {}}
        # Counterbalance the order without using row answers, then save both arms.
        order = ("native", "bf16") if index % 2 else ("bf16", "native")
        for arm in order:
            answer = enc.answer_hf(prefix, [q], native=arm == "native", top_logprobs=0)[0]
            if answer["prompt_tokens"] != len(direct.native_ids):
                raise ValueError(f"{row['id']} {arm} saw the wrong number of prompt tokens")
            item["arms"][arm] = {"text": answer["text"], "ids": answer["ids"],
                "finish": answer["finish_reason"], "prompt_tokens": answer["prompt_tokens"]}
        json_to(path, item)
        print("target", args.source, args.key, index, "/", len(rows), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("source", "target"))
    p.add_argument("key", choices=DEFAULT_MODELS)
    p.add_argument("--source", choices=DEFAULT_MODELS)
    p.add_argument("--cases", type=Path, default=HERE / "case-manifest.json")
    p.add_argument("--registry", type=Path, default=HERE / "models.json")
    p.add_argument("--weights", default="trained/translator")
    p.add_argument("--out", type=Path, default=Path("coding-work"))
    p.add_argument("--max-tokens", type=int, default=48)
    args = p.parse_args()
    if args.stage == "target" and (args.source is None or args.source == args.key):
        p.error("Target stage needs --source and a different receiver")
    if args.max_tokens < 1:
        p.error("Choose a positive output token cap")
    manifest = args.cases.read_bytes()
    rows = public_cases(manifest)
    registry = json.loads(args.registry.read_bytes())

    import torch
    from speculative_encoding import models

    for name in {args.key, args.source} - {None}:
        entry = registry[name]
        models.CATALOG[name] = dataclasses.replace(models.CATALOG[name], hf_id=entry["repo"], revision=entry["revision"])
    torch.set_num_threads(4)
    with torch.inference_mode():
        if args.stage == "source":
            source_stage(args, rows, torch, models)
        else:
            metadata = {"manifest_sha256": sha(manifest), "runner_sha256": sha(Path(__file__).read_bytes()),
                        "max_tokens": args.max_tokens, "source_revision": registry[args.source]["revision"],
                        "target_revision": registry[args.key]["revision"]}
            target_stage(args, rows, metadata, torch, models)


if __name__ == "__main__":
    main()
