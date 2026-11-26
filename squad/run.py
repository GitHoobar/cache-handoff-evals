"""Use the original translator with the same on-disk memories across targets."""

import argparse
import dataclasses
import json
from pathlib import Path
import random
import time

import torch
from speculative_encoding import models
from speculative_encoding.api import SpeculativeEncoder, MultiTargetEncoder
from speculative_encoding.encode import EncodedMemory
from codec import pack, unpack


def cases():
    articles = json.loads(Path("dev.json").read_bytes())["data"]
    rng = random.Random(16391)
    rng.shuffle(articles)
    by_article = [rng.sample(a["paragraphs"], 2) for a in articles]
    for i in range(64):
        article = articles[i % len(articles)]
        para = by_article[i % len(articles)][i // len(articles)]
        qa = rng.choice(para["qas"])
        yield dict(
            id=qa["id"],
            title=article["title"],
            document=para["context"],
            question=qa["question"],
            golds=[a["text"] for a in qa["answers"]],
        )


def measure(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def direct_model(key):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from speculative_encoding.translator.hf import ModelInterface

    entry = models.CATALOG[key]
    config = AutoConfig.from_pretrained(entry.hf_id, revision=entry.revision)
    cls = (
        AutoModelForImageTextToText
        if any(n.endswith("ForConditionalGeneration") for n in config.architectures)
        else AutoModelForCausalLM
    )
    model = (
        cls.from_pretrained(
            entry.hf_id,
            revision=entry.revision,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
    )
    return ModelInterface(model, key, check=False), models.load_tokenizer(key)


def prepare(a, rows):
    source = direct_model(a.source)
    mt = MultiTargetEncoder.from_pretrained(
        a.weights,
        a.source,
        [a.target],
        level=0,
        models={a.source: source},
        offload_targets=True,
        dtype=torch.bfloat16,
        chunk=getattr(a, "chunk", 32768),
    )
    mt.translator.set_compiled_attention(False)
    root = Path("memories") / a.source
    root.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        dest = root / (row["id"] + ".safetensors")
        if dest.exists():
            continue
        _, mem = mt.shared_input(row["document"], share="memory")
        mem.save(dest, dtype=torch.bfloat16)
        if (idx + 1) % 8 == 0:
            print("source", idx + 1, flush=True)


def decode(a, rows):
    model = direct_model(a.target)
    enc = SpeculativeEncoder.from_pretrained(
        a.weights,
        a.source,
        a.target,
        load_source=False,
        models={a.target: model},
        level=0,
        dtype=torch.bfloat16,
        chunk=getattr(a, "chunk", 32768),
    )
    enc.translator.set_compiled_attention(False)
    root = Path("memories") / a.source
    scratch = Path("scratch-" + a.target)
    scratch.mkdir(exist_ok=True)
    packed = scratch / "packed"
    restored = scratch / "restored"
    out = Path(f"answers-{a.source}-{a.target}.jsonl")
    seen = (
        {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    )
    for idx, row in enumerate(rows):
        if row["id"] in seen:
            continue
        src = root / (row["id"] + ".safetensors")
        base = EncodedMemory.load(src)
        q = enc.question(
            row["question"] + " Answer briefly using only the answer from the passage.",
            max_tokens=24,
        )
        entry = {k: v for k, v in row.items() if k != "document"}
        entry.update(tokens=base.tokens, shape=list(base.memory.shape), arms={})

        def answer(mem, native=False):
            prefix = enc.encode(memory=mem).full
            ans = enc.answer_hf(prefix, [q], native=native, top_logprobs=0)[0]
            return ans

        reference = answer(base)
        ref = reference["first_logprobs"].float()

        def record(ans, size, **other):
            lp = ans["first_logprobs"].float()
            valid = ref.isfinite() & lp.isfinite()
            kl = (ref[valid].exp() * (ref[valid] - lp[valid])).sum().item()
            return dict(text=ans["text"], ids=ans["ids"], bytes=size, first_kl=kl, **other)

        entry["arms"]["bf16"] = record(reference, src.stat().st_size)
        for label, bits, centered in [("int8", 8, False), ("int4", 4, False), ("center4", 4, True)]:
            _, pack_ms = measure(lambda: pack(src, packed, bits, centered))
            _, unpack_ms = measure(lambda: unpack(packed, restored))
            recovered = EncodedMemory.load(restored)
            assert (
                recovered.source_doc_ids == base.source_doc_ids
                and recovered.fingerprint == base.fingerprint
            )
            error = (
                torch.linalg.vector_norm(recovered.memory.float() - base.memory.float())
                / torch.linalg.vector_norm(base.memory.float())
            ).item()
            entry["arms"][label] = record(
                answer(recovered),
                packed.stat().st_size,
                pack_ms=pack_ms,
                unpack_ms=unpack_ms,
                relative_error=error,
            )
        entry["arms"]["native"] = record(answer(base, native=True), 0)
        other = EncodedMemory.load(root / (rows[(idx + 1) % len(rows)]["id"] + ".safetensors"))
        indices = torch.linspace(0, other.tokens - 1, base.tokens).round().long()
        wrong = dataclasses.replace(base, memory=other.memory[indices])
        entry["arms"]["wrong"] = record(answer(wrong), 0)
        with out.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        if (idx + 1) % 4 == 0:
            print("target", idx + 1, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["source", "target"])
    p.add_argument("--source", default="qwen3-14b")
    p.add_argument("--target", default="gemma3-12b")
    p.add_argument("--weights", default="trained/translator")
    a = p.parse_args()
    models.CATALOG["gemma3-12b"] = dataclasses.replace(
        models.CATALOG["gemma3-12b"],
        hf_id="unsloth/gemma-3-12b-it",
        revision="9478e665381f42974aa06177b019352fb6291876",
    )
    torch.set_num_threads(4)
    with torch.inference_mode():
        (prepare if a.stage == "source" else decode)(a, list(cases()))
