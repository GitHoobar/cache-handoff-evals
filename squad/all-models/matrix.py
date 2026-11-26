"""Build source files once and evaluate each receiver with the released HF pipeline."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from speculative_encoding import models
from speculative_encoding.api import MultiTargetEncoder, SpeculativeEncoder
from speculative_encoding.encode import EncodedMemory
from codec import make_reference, pack, load_packed
from run import cases

REGISTRY = json.loads(Path(__file__).with_name("models.json").read_text())
ARMS = ("bf16", "int8", "int4", "shared2")
PROBE = "This sentence initializes a reusable source model reference."
OUT = Path("matrix-results")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def configure():
    for key, spec in REGISTRY.items():
        models.CATALOG[key] = dataclasses.replace(models.CATALOG[key],
                    hf_id=spec["repo"], revision=spec["revision"])


def build_source(key, rows):
    pair = models.load_model(key, device="cuda", check=False)
    encoder = MultiTargetEncoder.from_pretrained("trained/translator", key, [key], level=0,
        dtype=torch.bfloat16, models={key:pair}, offload_targets=True)
    encoder.translator.set_compiled_attention(False)
    root = Path("memories") / key
    root.mkdir(parents=True, exist_ok=True)
    ref = root / "reference"
    _, probe = encoder.shared_input(PROBE, share="memory")
    probe.save(root / "probe", dtype=torch.bfloat16)
    make_reference(root / "probe", ref)
    inventory = {}
    for row in rows:
        folder = root / row["id"]
        folder.mkdir(exist_ok=True)
        _, mem = encoder.shared_input(row["document"], share="memory")
        base = folder / "bf16"
        mem.save(base, dtype=torch.bfloat16)
        for arm, bits in (("int8",8), ("int4",4), ("shared2",2)):
            pack(base, folder / arm, bits, reference=ref if arm=="shared2" else None)
        inventory[row["id"]] = {"tokens":mem.tokens, "fingerprint":mem.fingerprint,
            "bytes":{arm:(folder / arm).stat().st_size for arm in ARMS}}
    write_json(OUT / "sources" / (key+".json"), {"reference_bytes":ref.stat().st_size,"files":inventory})


def record(ans):
    return {"text":ans["text"], "ids":ans["ids"], "finish":ans["finish_reason"]}


def target_pair(model, src, tgt, rows):
    enc = SpeculativeEncoder.from_pretrained("trained/translator", src, tgt, level=0,
        load_source=False, dtype=torch.bfloat16, models={tgt:model})
    enc.translator.set_compiled_attention(False)
    root = Path("memories") / src
    for idx, row in enumerate(rows):
        output = OUT / "pairs" / (src+"__"+tgt) / (row["id"]+".json")
        if output.exists():
            continue
        q = enc.question(row["question"]+" Answer briefly using only the answer from the passage.",max_tokens=24)
        answers = {}
        for arm in ARMS:
            path = root / row["id"] / arm
            mem = EncodedMemory.load(path) if arm=="bf16" else load_packed(path, root/"reference" if arm=="shared2" else None)
            prefix = enc.encode(memory=mem).full
            ans = enc.answer_hf(prefix, [q], top_logprobs=0)[0]
            answers[arm] = record(ans)
            if arm == "bf16":
                reference_ids = prefix.prefix_ids
                native_path = OUT / "native" / tgt / (row["id"]+".json")
                digest = hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest()
                if not native_path.exists():
                    native = enc.answer_hf(prefix,[q],native=True,top_logprobs=0)[0]
                    write_json(native_path, {**record(native),"prefix_sha256":digest})
                elif json.loads(native_path.read_bytes())["prefix_sha256"] != digest:
                    raise ValueError("native prefix IDs changed for target on "+row["id"])
            else:
                assert prefix.prefix_ids == reference_ids
        entry = {k:v for k,v in row.items() if k!="document"}
        entry.update(source=src, target=tgt, arms=answers)
        if idx % 16 == 0:
            repeated = enc.answer_hf(enc.encode(memory=EncodedMemory.load(root/row["id"]/"bf16")).full,
                                     [q],top_logprobs=0)[0]
            entry["repeat_bf16_equal"] = answers["bf16"]["ids"] == repeated["ids"]
        write_json(output,entry)
    print("complete",src,tgt,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("source","target"))
    parser.add_argument("key", choices=list(REGISTRY))
    parser.add_argument("--sources", nargs="+", choices=list(REGISTRY))
    args = parser.parse_args()
    configure()
    torch.set_num_threads(4)
    with torch.inference_mode():
        rows = list(cases())
        if args.stage == "source":
            build_source(args.key,rows)
        else:
            model = models.load_model(args.key,device="cuda",check=False)
            for src in args.sources or list(REGISTRY):
                if src!=args.key and not all((OUT/"pairs"/(src+"__"+args.key)/(r["id"]+".json")).exists() for r in rows):
                    target_pair(model,src,args.key,rows)
