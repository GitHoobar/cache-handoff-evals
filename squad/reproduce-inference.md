# Regenerate the SQuAD handoffs on a GPU

Start at the repository root on a Linux machine with a CUDA GPU that holds a BF16 14B model and the translator. The original runs used NVIDIA L40S GPUs with 48 GB of memory. Allow about 220 GB of free disk space for model downloads and intermediate files. The included scripts are the original seven-model experiment drivers. They generate native answers and four translated-memory conditions, including the released BF16 condition used for the country-lookup analysis.

Use a Python environment with a CUDA-enabled PyTorch installation. The original inference environment recorded Python 3.13.18 and PyTorch 2.12.1+cu130. The commands below keep the installed PyTorch build and import the translator from its pinned source checkout. Model downloads use the revisions in [models.json](all-models/models.json) and translator revision `eec4d43036a15035817418be2e4591b8e42a06cf`.

```bash
set -eu
squad_scripts="$(cd squad && pwd)"
work_dir="$(mktemp -d "$PWD/../squad-inference.XXXXXX")"
cd "$work_dir"
export HF_HOME="$work_dir/hf-cache"
export TOKENIZERS_PARALLELISM=false
python3 -c 'import torch; print(torch.__version__); assert torch.cuda.is_available()'
python3 -m pip install 'transformers==5.23.0' 'huggingface-hub==1.33.0' 'accelerate==1.16.0' 'safetensors==0.8.0' 'sentencepiece==0.2.3'
git clone https://github.com/b10-tv/speculative-encoding.git release
git -C release checkout 5b74ad4d14e815833b8950d040960e4f5fdf3430
export PYTHONPATH="$work_dir/release"
curl -fL https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o dev.json
printf '%s  dev.json\n' 95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9 | sha256sum --check
HF_HUB_DISABLE_XET=1 python3 "$squad_scripts/all-models/fetch_matrix.py"
export HF_HUB_OFFLINE=1

models=(qwen3-14b gemma3-12b ministral3-8b llama31-8b gemma4-12b qwen35-9b ministral3-14b)
for model in "${models[@]}"; do
    python3 "$squad_scripts/all-models/matrix.py" source "$model"
done
for model in "${models[@]}"; do
    python3 "$squad_scripts/all-models/matrix.py" target "$model"
done
python3 "$squad_scripts/all-models/analyze_matrix.py" matrix-results fresh-summary.json
tar -czf fresh-matrix-results.tar.gz matrix-results
python3 "$squad_scripts/replay_lookup.py" fresh-matrix-results.tar.gz --output fresh-lookup.json
```

[run.py](run.py) selects the 64 questions from the official dev file with seed 16391. [matrix.py](all-models/matrix.py) builds each source memory and tests all six other senders for each receiver. The working directory now contains the regenerated outputs in `matrix-results/`. Compare the `summary` field of `fresh-lookup.json` with the [archived lookup](lookup-results.json). The archive digest covers the exact tarball bytes, including packaging metadata. The original distributed run also repeated all 64 native Qwen 3 14B answers on a second machine. The single-GPU workflow above covers the 42 directed comparisons and the scheduled BF16 repeat checks.
