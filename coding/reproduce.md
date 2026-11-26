# Reproduce the coding handoff experiment

This bundle contains original miniature Python repositories, all prompts, raw first and second responses, individual test observations and the scoring programs. Three frozen receiver models answer the same question under native prefill and the released BF16 cache translator. The receiver can see the document through the translator's normal interface. All tools change at most one named setting in a disposable authored repository.

## Check the experiment without a GPU

Python 3.10 or newer is sufficient for all numerical checks. Run from the extracted directory.

```sh
python3 cases.py --output regenerated-cases.json
cmp regenerated-cases.json case-manifest.json
python3 make_panels.py --cases regenerated-cases.json --output regenerated-panels.json
cmp regenerated-panels.json coding-panels.json
python3 score_coding.py predictions.tar.gz --output checked.json
cmp checked.json coding-results.json
python3 score_feedback.py predictions.tar.gz --output checked-feedback.json
cmp checked-feedback.json feedback-results.json
python3 score_toolmode.py predictions.tar.gz --output checked-structured.json
cmp checked-structured.json toolmode-results.json
python3 diagnostics.py coding-results.json --output checked-diagnostics.json
cmp checked-diagnostics.json diagnostics.json
python3 verify_repeats.py
python3 replay_fallback.py
```

The repeat verifier reads both archives, recomputes the selection from all primary cases and checks the original and repeated input metadata, exact generated tokens, and recorded summaries. It can compare the recorded donor tensor digests. Regenerating those tensors requires the GPU inference procedure below.

To apply one archived generation as a physical edit in a temporary directory, choose a case, route and arm. The replay prints the exact model response, the settings diff and the behavior before and after the edit. It runs the original fixture code in new Python processes. The only permitted model-controlled value is a pre-existing constant name.

```sh
python3 replay_action.py predictions.tar.gz --case batch-writer-a --source ministral3-14b --target qwen3-14b --arm native
python3 replay_action.py predictions.tar.gz --case batch-writer-a --source ministral3-14b --target qwen3-14b --arm bf16
```

Add `--mode structured` to replay the model's constrained JSON choice. The finite grammar permits exactly the two canonical JSON argument objects. We let the model select the higher-logit branch at the first token where the two completions diverge. The remaining tokens are forced by the chosen branch. This measures a particular grammar and serialization of the two actions. [Structured decoding protocol](toolmode-protocol.md)

## Repeat inference

We used the Hugging Face backend with the released translator in eager BF16, greedy generation, a 48-token cap per action, and an NVIDIA RTX PRO 6000 Blackwell Server Edition. The source and receiver are loaded in separate stages on one GPU. The model registry pins the official Qwen and Ministral checkpoints and a Gemma 3 mirror with matching model weight objects and documented metadata differences. Download models before enabling offline mode.

```sh
git clone https://github.com/b10-tv/speculative-encoding.git release
git -C release checkout 5b74ad4d14e815833b8950d040960e4f5fdf3430
# In the AWS PyTorch environment described below, using its Python interpreter
python3 -m pip install 'transformers==5.23.0' 'huggingface-hub==1.33.0' 'accelerate==1.16.0' 'safetensors==0.8.0' 'sentencepiece==0.2.3'
export PYTHONPATH="$PWD/release"
export TOKENIZERS_PARALLELISM=false
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from fetch_matrix import fetch
for key in ("translator", "qwen3-14b", "ministral3-14b", "gemma3-12b"):
    fetch(key)
PY
export HF_HUB_OFFLINE=1
for model in qwen3-14b ministral3-14b gemma3-12b; do
    python3 run_coding.py source "$model" --cases coding-panels.json --out coding-work
done
for target in ministral3-14b qwen3-14b gemma3-12b; do
    for source in qwen3-14b ministral3-14b gemma3-12b; do
        if [ "$target" != "$source" ]; then
            python3 run_coding.py target "$target" --source "$source" --cases coding-panels.json --out coding-work
        fi
    done
done
python3 run_feedback.py target qwen3-14b --source ministral3-14b --cases coding-panels.json --out coding-work
python3 run_feedback.py target ministral3-14b --source qwen3-14b --cases coding-panels.json --out coding-work
python3 repeat_coding.py --cases coding-panels.json --out coding-work --repeat-work coding-repeat
for target in ministral3-14b qwen3-14b gemma3-12b; do
    for source in qwen3-14b ministral3-14b gemma3-12b; do
        if [ "$target" != "$source" ]; then
            python3 run_toolmode.py target "$target" --source "$source" --cases coding-panels.json --out coding-work
        fi
    done
done
python3 score_coding.py coding-work --output fresh-scored.json
python3 score_feedback.py coding-work --output fresh-feedback.json
python3 score_toolmode.py coding-work --output fresh-structured.json
```

The machine used the AWS Deep Learning OSS PyTorch 2.12 Ubuntu 24.04 image. Its PyTorch environment ran Python 3.13.18, Torch 2.12.1+cu130, NumPy 2.3.5, tokenizers 0.23.3 and the package versions shown above. The translator revision is `eec4d43036a15035817418be2e4591b8e42a06cf`. [Checkpoint](https://huggingface.co/RampPublic/speculative-encoding/tree/eec4d43036a15035817418be2e4591b8e42a06cf/translator) · [Inference software](https://github.com/b10-tv/speculative-encoding/tree/5b74ad4d14e815833b8950d040960e4f5fdf3430) · [Original study](https://labs.baseten.co/articles/speculative-encoding-many-to-many-cache-translation)
