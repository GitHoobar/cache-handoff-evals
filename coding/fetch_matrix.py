"""Download only the pinned HF weights, configs, tokenizers and translator."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = json.loads(Path(__file__).with_name("models.json").read_text())
REV = "eec4d43036a15035817418be2e4591b8e42a06cf"


def fetch(key):
    if key == "translator":
        snapshot_download("RampPublic/speculative-encoding", revision=REV, local_dir="trained",
                          allow_patterns=["translator/*.pt", "translator/meta.json"], max_workers=5)
    else:
        entry = MODELS[key]
        snapshot_download(entry["repo"], revision=entry["revision"], max_workers=4,
                          allow_patterns=[*entry["weight_git_objects"], "*.json", "*.jinja",
                                          "*.model", "merges.txt", "vocab.txt"])
    Path("ready").mkdir(exist_ok=True)
    Path("ready", key).touch()
    print("downloaded", key, flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, key): key for key in ["translator", *MODELS]}
        for job in as_completed(futures):
            job.result()
