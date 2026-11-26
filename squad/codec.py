"""Pack an upstream BF16 EncodedMemory or restore an ordinary upstream file."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "speculative-encoding-memory/1"
PACKED = "experimental-shared-memory-quantized/1"
GROUP = 64


def make_reference(src, dst):
    """Save one source-specific offset for use on unrelated documents."""
    with safe_open(str(src), framework="pt") as handle:
        meta = handle.metadata()
        if meta.get("format") != FORMAT:
            raise ValueError("expected an upstream memory file")
        info = json.loads(meta["info"])
        x = handle.get_tensor("memory")
        center = x.mean(0, dtype=torch.float32).bfloat16()
    save_file(
        {"center": center},
        str(dst),
        metadata={
            "source": info["source"],
            "fingerprint": info["fingerprint"],
            "format": "shared-offset/1",
        },
    )


def _reference(path, info, shape):
    if path is None:
        raise ValueError("this packed memory needs its shared reference file")
    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata()
        if meta.get("format") != "shared-offset/1" or any(
            meta.get(k) != info[k] for k in ("source", "fingerprint")
        ):
            raise ValueError("reference and memory belong to different encoders")
        center = handle.get_tensor("center")
    if tuple(center.shape) != tuple(shape) or center.dtype != torch.bfloat16:
        raise ValueError("invalid reference geometry or dtype")
    digest = hashlib.sha256(center.view(torch.int16).numpy().tobytes()).hexdigest()
    return center, digest


def pack(src, dst, bits, centered=False, reference=None):
    if bits not in (2, 4, 8):
        raise ValueError("supported widths are 2, 4 and 8")
    with safe_open(str(src), framework="pt") as handle:
        header = handle.metadata()
        if header.get("format") != FORMAT:
            raise ValueError("expected an upstream memory file")
        mem = handle.get_tensor("memory")
        if mem.dtype != torch.bfloat16 or mem.ndim != 3 or not all(mem.shape):
            raise ValueError("expected a nonempty 3D BF16 memory")
        tokens, streams, width = mem.shape
        groups = (width + GROUP - 1) // GROUP
        padded = groups * GROUP
        limit = 2 ** (bits - 1) - 1
        center = None
        options = dict(bits=bits, width=width, group=GROUP, centered=centered)
        if reference is not None:
            if centered:
                raise ValueError("choose a document mean or a shared reference")
            center, options["reference"] = _reference(
                reference, json.loads(header["info"]), (streams, width)
            )
        elif centered:
            sums = torch.zeros((streams, width), dtype=torch.float32)
            for slab in mem.split(32):
                sums.add_(slab.sum(dim=0, dtype=torch.float32))
            center = (sums / tokens).bfloat16()
        scales = torch.empty((tokens, streams, groups), dtype=torch.bfloat16)
        codes = torch.empty(
            (tokens, streams, padded * bits // 8), dtype=torch.int8 if bits == 8 else torch.uint8
        )
        for start in range(0, tokens, 32):
            vals = mem[start : start + 32].float()
            if center is not None:
                vals -= center
            if padded != width:
                vals = torch.nn.functional.pad(vals, (0, padded - width))
            vals = vals.reshape(len(vals), streams, groups, GROUP)
            scale = (
                (vals.square().mean(-1).sqrt()) if bits == 2 else (vals.abs().amax(-1) / limit)
            ).bfloat16()
            scales[start : start + len(vals)] = scale
            divisor = torch.where(scale == 0, 1.0, scale.float()).unsqueeze(-1)
            normalized = vals / divisor
            if bits == 2:
                # Four symmetric Lloyd-Max reconstruction levels for a unit Gaussian.
                flat = normalized.flatten(-2)
                symbols = (flat >= -0.9816).byte() + (flat >= 0).byte() + (flat >= 0.9816).byte()
                codes[start : start + len(vals)] = (
                    symbols[..., 0::4]
                    | (symbols[..., 1::4] << 2)
                    | (symbols[..., 2::4] << 4)
                    | (symbols[..., 3::4] << 6)
                )
                continue
            indices = normalized.round().clamp(-limit, limit).to(torch.int8).flatten(-2)
            if bits == 4:
                pos = (indices + limit).to(torch.uint8)
                indices = pos[..., ::2] | (pos[..., 1::2] << 4)
            codes[start : start + len(vals)] = indices
        tensors = {"codes": codes, "scales": scales}
        if centered:
            tensors["center"] = center
        save_file(
            tensors, str(dst), metadata={**header, "format": PACKED, "codec": json.dumps(options)}
        )


def _restore(src, reference=None):
    with safe_open(str(src), framework="pt") as handle:
        header = handle.metadata()
        if header.get("format") != PACKED:
            raise ValueError("expected a packed memory file")
        options = json.loads(header.pop("codec"))
        bits, width = options["bits"], options["width"]
        codes, scales = handle.get_tensor("codes"), handle.get_tensor("scales")
        center = handle.get_tensor("center") if options["centered"] else None
        tokens, streams, groups = scales.shape
        if "reference" in options:
            center, digest = _reference(reference, json.loads(header["info"]), (streams, width))
            if digest != options["reference"]:
                raise ValueError("incorrect reference contents")
        if (
            bits not in (2, 4, 8)
            or options["group"] != GROUP
            or groups != (width + GROUP - 1) // GROUP
            or width < 1
            or codes.shape != (tokens, streams, groups * GROUP * bits // 8)
        ):
            raise ValueError("invalid packed geometry")
        mem = torch.empty((tokens, streams, width), dtype=torch.bfloat16)
        for start in range(0, tokens, 32):
            block = codes[start : start + 32]
            if bits == 2:
                symbols = torch.stack(tuple((block >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
                levels = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])
                block = levels[symbols.long()].flatten(-2)
            elif bits == 4:
                block = torch.stack((block & 15, block >> 4), dim=-1).flatten(-2).to(torch.int8) - 7
            vals = (
                block.reshape(len(block), streams, groups, GROUP).float()
                * scales[start : start + len(block)].unsqueeze(-1)
            ).flatten(-2)[..., :width]
            if center is not None:
                vals += center
            mem[start : start + len(block)] = vals
        return mem, {**header, "format": FORMAT}


def load_packed(src, reference=None):
    """Return a normal upstream EncodedMemory directly from the packed file."""
    from speculative_encoding.encode import EncodedMemory

    tensor, header = _restore(src, reference)
    info = json.loads(header["info"])
    info["source_doc_spans"] = [tuple(span) for span in info["source_doc_spans"]]
    return EncodedMemory(memory=tensor, **info)


def unpack(src, dst, reference=None):
    tensor, header = _restore(src, reference)
    save_file({"memory": tensor}, str(dst), metadata=header)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("reference", "pack", "unpack"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bits", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--center", action="store_true")
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve():
        p.error("choose different input and output files")
    if args.action == "reference":
        make_reference(args.input, args.output)
    elif args.action == "pack":
        pack(args.input, args.output, args.bits, args.center, args.reference)
    else:
        unpack(args.input, args.output, args.reference)
