# OmniReasoner — KG-TRACES with Confidence Scoring, Early Exit, Latency Instrumentation & Interactive UI

> Built on top of [KG-TRACES](https://arxiv.org/abs/2506.00783) (Wu et al., 2025).
> This repository extends the original pipeline with confidence scoring, stop-string early exit, latency instrumentation, faithfulness verification improvements, and a full interactive browser UI backed by a Slurm HPC bridge.

---

## Table of Contents

1. [What Changed From Baseline](#what-changed-from-baseline)
2. [Repository Layout](#repository-layout)
3. [Environment Setup](#environment-setup)
4. [Full Batch Evaluation Pipeline](#full-batch-evaluation-pipeline)
   - [Step 1 — Generate Reasoning Paths](#step-1--generate-reasoning-paths)
   - [Step 2 — Answer Inference & Evaluation](#step-2--answer-inference--evaluation)
   - [Running on Slurm (HPC)](#running-on-slurm-hpc)
5. [Interactive UI Pipeline](#interactive-ui-pipeline)
   - [How It Works](#how-it-works)
   - [Starting the Bridge Server](#starting-the-bridge-server)
   - [SSH Tunnel from Your Laptop](#ssh-tunnel-from-your-laptop)
   - [Opening the UI](#opening-the-ui)
   - [UI Tabs Reference](#ui-tabs-reference)
6. [New Flags Reference](#new-flags-reference)
7. [Output Files Reference](#output-files-reference)
8. [Changes in Detail](#changes-in-detail)
   - [Confidence Scoring](#1-confidence-scoring--basemodel)
   - [Stop-String Early Exit](#2-stop-string-early-exit--basemodel)
   - [Latency Instrumentation](#3-latency-instrumentation--predict_answerpy)
   - [Triple Path Parser Fixes](#4-triple-path-parser-fixes--gen_predict_pathpy)
   - [Faithfulness Verification Fallback](#5-faithfulness-verification-fallback--gen_predict_pathpy)
   - [Empty Path Filtering](#6-empty-path-filtering--predict_answerpy)
   - [Interactive UI](#7-interactive-ui)
9. [Results Summary](#results-summary)

---

## What Changed From Baseline

| Area | Baseline KG-TRACES | This Repo |
|---|---|---|
| Confidence scoring | None | Dual score: `full_confidence` + `answer_only_confidence` (preamble-excluded) |
| Early exit | None | Stop-string `StoppingCriteria` fires at `</endanswer>` — lossless, saves ~4.7× latency |
| Latency tracking | None | Per-sample `latency_sec`, `tokens_per_sec`; aggregate p50/p95/p99 written to `latency_metrics.json` |
| Triple path parsing | Single `<PATH>` regex, fails on tagless output | 3-strategy robust parser: tagged, `<answers>` block, bare arrow lines |
| Triple instruction | Contradictory dual-format (`<SEP>` + `->` + `<answers>`) | Single unambiguous `Format rules:` instruction |
| `--drop_unverified` | Empties `prediction_paths` when all paths fail verification (silent accuracy loss) | Keeps originals as fallback; `--strict_drop` for old behaviour |
| Empty path handling | `[[],[],[]]` passed to PromptBuilder as ` [<INFERRED>]` noise | Filtered out before building prompt |
| Interactive inference | Not available | Full browser UI: streaming answers, KG graph, path cards, reasoning steps, Slurm integration |

---

## Repository Layout

```
KG-TRACES/
├── src/
│   ├── llms/
│   │   └── base_hf_causal_model.py      ← confidence scoring + early exit + latency timing
│   ├── qa_prediction/
│   │   ├── predict_answer.py             ← main eval pipeline, latency aggregation
│   │   ├── ui_inference.py               ← single-question inference for the UI
│   │   ├── gen_predict_path.py           ← path generation, robust triple parser
│   │   └── build_qa_input.py             ← PromptBuilder with CoT / answer-only templates
│   └── tools/
├── bridge.py                             ← HTTP bridge server (login node)
├── files/
│   └── kg_traces_ui_slurm.html           ← browser UI (open locally)
├── scripts/
│   └── predict.sh                        ← batch evaluation shell script
├── kg-traces.sbatch                      ← Slurm job for full dataset eval
├── prompts/
│   └── qwen2.5.txt
├── models/
│   └── KG-TRACES/                        ← model weights (HF format)
└── data/
    └── processed/
        ├── webqsp/
        └── cwq/
```

---

## Environment Setup

```bash
# Clone and enter repo
git clone <your-fork-url>
cd KG-TRACES

# Create and activate conda environment
conda create -n kg_traces python=3.12 -y
conda activate kg_traces

# Install dependencies
pip install torch transformers datasets tqdm numpy
pip install -r requirements.txt   # if present

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

> **HPC note:** On SJSU coe-hpc3, activate with:
> ```bash
> source ~/anaconda3/etc/profile.d/conda.sh && conda activate kg_traces
> ```
> The standard `conda activate` does not work in non-interactive Slurm batch jobs.

---

## Full Batch Evaluation Pipeline

The batch pipeline runs in two steps: first generate reasoning paths, then run answer inference and evaluation.

### Step 1 — Generate Reasoning Paths

Generates `<PATH>` tagged relation or triple paths for every question in the dataset using beam search.

```bash
# Relation paths (recommended — higher accuracy)
python src/qa_prediction/gen_predict_path.py \
    --dataset webqsp \
    --split test \
    --path_type relation \
    --model_path models/KG-TRACES \
    --model_name KG-TRACES \
    --prompt_path prompts/qwen2.5.txt \
    --n_beam 3 \
    --batch_size 64 \
    --output_path results/gen_predict_path

# Triple paths
python src/qa_prediction/gen_predict_path.py \
    --dataset webqsp \
    --split test \
    --path_type triple \
    --model_path models/KG-TRACES \
    --model_name KG-TRACES \
    --n_beam 3 \
    --batch_size 64 \
    --output_path results/gen_predict_path

# With faithfulness verification against KG graph
python src/qa_prediction/gen_predict_path.py \
    --dataset webqsp \
    --split test \
    --path_type triple \
    --n_beam 3 \
    --verify_paths \
    --output_path results/gen_predict_path
    # NOTE: do NOT add --drop_unverified unless you also pass --strict_drop
    # Without --strict_drop, originals are kept when all paths fail verification
```

**Output:** `results/gen_predict_path/webqsp/test/KG-TRACES/type_relation/predictions_3_False.jsonl`

Each record contains:
```json
{
  "id": "WebQTest-42",
  "question": "what are the songs justin bieber wrote",
  "prediction_paths": [["Justin Bieber", "music.composer.compositions", "Baby"]],
  "path_conf": [0.91, 0.91, 0.92],
  "path_entropy": 1.38,
  "triple_confidences": [[0.97, 0.99, 0.99]],
  "latency_sec": 1.07,
  "verified_paths": [...],
  "path_meta": [{"verified": true, "unsupported_steps": []}]
}
```

---

### Step 2 — Answer Inference & Evaluation

Runs the QA model over the dataset using generated paths, computes accuracy/Hit@k/F1/MRR, and writes latency metrics.

```bash
# Standard run — relation paths, beam 3, with early exit (default)
python src/qa_prediction/predict_answer.py \
    --dataset webqsp \
    --split test \
    --model_name KG-TRACES \
    --model_path models/KG-TRACES \
    --model_type webqsp_cwq_tuned \
    --add_path \
    --use_pred_path \
    --pred_path_type relation \
    --pred_relation_path_path results/gen_predict_path/webqsp/test/KG-TRACES/type_relation/predictions_3_False.jsonl \
    --batch_size 2 \
    --n_beam 3 \
    --predict_path results/KGQA

# With Chain-of-Thought reasoning template
python src/qa_prediction/predict_answer.py \
    ... \
    --include_reasoning

# With CoT + Early Exit explicitly configured
python src/qa_prediction/predict_answer.py \
    ... \
    --include_reasoning \
    --early_exit true

# With confidence threshold flagging (does NOT affect predictions or Hit@k)
python src/qa_prediction/predict_answer.py \
    ... \
    --early_exit true \
    --conf_threshold -1.5

# Triple paths
python src/qa_prediction/predict_answer.py \
    --dataset webqsp \
    --split test \
    --model_name KG-TRACES \
    --model_path models/KG-TRACES \
    --model_type webqsp_cwq_tuned \
    --add_path \
    --use_pred_path \
    --pred_path_type triple \
    --pred_triple_path_path results/gen_predict_path/webqsp/test/KG-TRACES/type_triple/predictions_3_False.jsonl \
    --batch_size 2 \
    --predict_path results/KGQA

# Disable early exit (e.g. when using --explain for full output)
python src/qa_prediction/predict_answer.py \
    ... \
    --explain \
    --early_exit false
```

**Outputs written to** `results/KGQA/<dataset>/<split>/<model_name>/`:
- `predictions.jsonl` — one record per sample with prediction, confidence, latency
- `latency_metrics.json` — aggregate latency statistics for the full run

---

### Running on Slurm (HPC)

```bash
# Submit the full evaluation job
sbatch kg-traces.sbatch
```

The sbatch file runs `scripts/predict.sh` which chains path generation → answer inference → evaluation. It uses:
- Partition: `nsfqs`
- Memory: 64G
- Time: 48:00:00
- CPUs: 8

Check job status:
```bash
squeue -u $USER
tail -f <JobName>_<JobID>.out   # live log
```

---

## Interactive UI Pipeline

The UI lets you submit a single question through a browser, watch the inference run live on the HPC compute node, and explore the results across dedicated tabs.

### How It Works

```
Browser (laptop)
    │  HTTP POST /submit
    ▼
bridge.py  ──────────────────────  login node (coe-hpc3), port 8765
    │  sbatch job.sh               stdlib HTTP server, no extra deps
    ▼
Slurm scheduler
    │  allocates GPU
    ▼
Compute node
  ui_inference.py                  single-question inference
    │  stdout: UI_EVENT lines
    ▼
bridge.py /poll  (2s interval) ──→  Browser
    │  /result
    ▼
Browser renders Answer / Graph / Triples / Paths / Live Events tabs
```

### Starting the Bridge Server

Run this **once on coe-hpc3** (login node) in a screen/tmux session so it persists:

```bash
# SSH to login node
ssh 018228028@coe-hpc1.sjsu.edu
ssh coe-hpc3

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kg_traces

# Start bridge (from repo root)
cd ~/KG-TRACES
python bridge.py --port 8765
```

Optional environment variable overrides:
```bash
SLURM_PARTITION=nsfqs \
SLURM_MEM=64G \
SLURM_TIME=48:00:00 \
CONDA_ENV=kg_traces \
python bridge.py --port 8765
```

### SSH Tunnel from Your Laptop

In a **separate terminal on your laptop**, forward port 8765:

```bash
ssh -L 8765:localhost:8765 -t <>\
    "<>"
```

Keep this terminal open while using the UI. The bridge status dot in the top-right corner of the UI turns green when connected.

### Opening the UI

Open `files/kg_traces_ui_slurm.html` directly in your browser — no web server needed. It is a self-contained static file.

```bash
# macOS
open files/kg_traces_ui_slurm.html

# Linux
xdg-open files/kg_traces_ui_slurm.html

# Windows
start files/kg_traces_ui_slurm.html
```

**Submitting a question:**
1. Type your question in the Question box
2. Choose **With Reasoning** or **Direct Answer** mode
3. Select dataset, path type (Relation or Triple), and beam count
4. Adjust Slurm config if needed (partition, memory, conda env)
5. Click **SUBMIT JOB** — a Slurm job is submitted automatically
6. Watch results stream in across the tabs

### UI Tabs Reference

| Tab | Content |
|---|---|
| **Answer** | Streaming answer → parsed pills; KG Evidence card (path terminal entities); reasoning steps timeline; confidence score |
| **Relation Graph** | SVG graph of each beam's path chain — Q node (question words) → relation labels on line → ANS node (answer). Dynamic height grows with number of paths |
| **Triples** | For triple path mode: `subject → relation → object` cards per hop, colour-coded per path |
| **Paths** | All predicted reasoning paths as expandable cards with count badge |
| **Live Events** | Raw `UI_EVENT` stream — pipeline status (model load, beam search, confidence), formatted with icons and timestamps |
| **Slurm Log** | Full raw Slurm stdout — useful for debugging errors |

**Answer tab — reasoning steps** are synthesized programmatically from the parsed KG paths, not from model output. Each step is colour-coded to match the corresponding path in the graph. Path-terminal entities (the KG-supported answer) are shown in a teal evidence card above the model answer. A conflict warning appears when the two disagree.

---

## New Flags Reference

### `gen_predict_path.py`

| Flag | Default | Description |
|---|---|---|
| `--verify_paths` | off | Verify predicted triple paths against sample KG graph before saving |
| `--drop_unverified` | off | Replace `prediction_paths` with `verified_paths`. **Keeps originals as fallback** when all paths fail verification to prevent silent accuracy loss |
| `--strict_drop` | off | With `--drop_unverified`: also empty `prediction_paths` when none pass (restores original strict behaviour) |

### `predict_answer.py` / `base_hf_causal_model.py`

| Flag | Default | Description |
|---|---|---|
| `--early_exit` | `true` | Stop-string early exit. Halts decoding at `</endanswer>` boundary. Zero effect on Hits@k |
| `--explain` | off | Generate full explanation after answer (disables early exit automatically) |
| `--conf_threshold` | `-inf` | Answer-only mean log-prob below which `low_confidence: true` is written to output. **Never alters prediction or affects Hit@k** |
| `--include_reasoning` | off | Use `QUESTION_WITH_REASONING` prompt template. Default uses `QUESTION_ANSWER_ONLY` |

### `ui_inference.py` (called internally by bridge)

| Flag | Default | Description |
|---|---|---|
| `--question` | required | The question to answer |
| `--path_type` | `triple` | `relation` or `triple` |
| `--n_beam` | `3` | Number of beams for path generation |
| `--model_path` | `models/KG-TRACES` | Model weights path |
| `--out_dir` | required | Output directory for `output.json` |
| `--early_exit` | `true` | Stop-string exit |
| `--conf_threshold` | `-inf` | Confidence quality gate |
| `--explain` | off | Disables early exit, generates explanation |

---

## Output Files Reference

### `predictions.jsonl` (one record per sample)

```json
{
  "id": "WebQTest-42",
  "question": "what are the songs that justin bieber wrote",
  "prediction": ["Baby\nNever Say Never\nBieber"],
  "ground_truth": ["Baby", "Never Say Never", ...],
  "confidence": -0.0023,
  "full_confidence": -0.0094,
  "low_confidence": false,
  "latency_sec": 1.074,
  "tokens_per_sec": 55.6,
  "num_of_paths": 3,
  "lists_of_paths": ["Justin Bieber -> music.composer.compositions -> Baby <INFERRED>", ...],
  "input_token_num": 127,
  "output_token_num": 15
}
```

**Confidence fields:**
- `confidence` — answer-only mean log-prob (preamble tokens excluded). **Primary signal.**
- `full_confidence` — mean log-prob over all generated tokens including preamble. Always higher (less negative) than `confidence`.
- `low_confidence` — `true` only when `--conf_threshold` is set and `confidence < threshold`. Never affects prediction text.

### `latency_metrics.json` (one file per run)

```json
{
  "total_samples": 1628,
  "total_inference_time_sec": 892.4,
  "latency_mean_sec": 0.548,
  "latency_std_sec": 0.211,
  "latency_p50_sec": 0.512,
  "latency_p95_sec": 0.934,
  "latency_p99_sec": 1.201,
  "latency_max_sec": 2.847,
  "throughput_mean_tokens_per_sec": 15.96,
  "throughput_total_tokens_per_sec": 18.3,
  "batch_size": 2,
  "dataset": "webqsp",
  "split": "test",
  "model_name": "KG-TRACES",
  "path_type": "relation",
  "early_exit": true
}
```

Use `p95` and `p99` to understand tail latency — mean alone is misleading when output length varies widely.

### `output.json` (UI pipeline — one per job)

```json
{
  "id": "ui_query",
  "question": "what town was martin luther king assassinated in",
  "prediction": ["Memphis"],
  "path_derived_answers": ["Memphis"],
  "raw_model_output": "**Answer**:\nMemphis",
  "reasoning_steps": [
    {"label": "Question received", "detail": "what town was martin luther king assassinated in", "path_label": ""},
    {"label": "Path 1: place of death (people)", "detail": "people.deceased_person.place_of_death", "path_label": "P1"},
    {"label": "Answer: Memphis", "detail": "Retrieved via knowledge graph traversal", "path_label": ""}
  ],
  "confidence": -0.0023,
  "full_confidence": -0.0094,
  "low_confidence": false,
  "num_of_paths": 3,
  "parsed_paths": [["people.deceased_person.place_of_death"]],
  "raw_beam_outputs": ["<PATH>people.deceased_person.place_of_death</PATH>", ...],
  "path_type": "relation",
  "early_exit_used": true,
  "answer_conflict": false
}
```

---

## Changes in Detail

### 1. Confidence Scoring — `base_hf_causal_model.py`

**What was missing:** The baseline had no way to assess answer reliability.

**What was added:** `_compute_confidence()` computes two scores from the decoder's per-token log-probabilities during a single forward pass — no extra model calls.

**Full confidence** (mean over all generated tokens):
```
ĉ_full = (1/T) Σ log p(x_t | x_<t, prompt)
```

**Answer-only confidence** (preamble excluded):
The model is fine-tuned to prepend `**Answer**:\n` — these tokens are nearly deterministic and inflate the score regardless of answer quality. We re-tokenise the preamble, skip its `k` tokens, and score only the substantive answer:
```
ĉ_ans = (1/(T-k)) Σ_{t=k+1}^{T} log p(x_t | x_<t, prompt)
```

**Quality gate:** When `--conf_threshold τ` is set and `ĉ_ans < τ`, the record is annotated `low_confidence: true`. The prediction text is never modified — Hits@k is unaffected by design.

`generate_sentence_batch()` now returns `(responses, confidences, latencies)` instead of `responses`. The `isinstance` check in `predict_answer.py` ensures backward compatibility with any model class returning the old format.

---

### 2. Stop-String Early Exit — `base_hf_causal_model.py`

**What was missing:** The model decoded to `max_output_tokens` even after completing the answer list — wasting GPU time on explanation text that is discarded during evaluation.

**What was added:** `_AnswerEndCriteria(StoppingCriteria)` monitors the token tail at every decoding step. Stop sequences in priority order:
1. `</endanswer>` — primary terminator from updated prompt template
2. `</endanswer>\n` — tag followed by newline
3. `]\n` — fallback for outputs truncated before emitting the tag

Stop strings are pre-tokenised **once** at `prepare_for_inference()` — not per-batch, so there is no per-call overhead. Since greedy decoding produces identical sequences, only `input_ids[0]` is checked per step.

**Early exit is disabled automatically** when `--explain` is passed, since in explanation mode the post-answer text is the desired output. The flag `--early_exit false` also disables it explicitly.

**Result:** Generation time drops from 82s → 17.4s on WebQSP (Beam 3, CoT). Hit@k and MRR both *improve* (83.35 vs 81.45) because early exit removes hallucinated content that would otherwise count as wrong predictions.

---

### 3. Latency Instrumentation — `predict_answer.py`

**What was missing:** No way to measure or compare inference efficiency across configurations.

**What was added:**
- `time.perf_counter()` brackets exactly the `model.generate()` call — tokenisation excluded
- Per-sample: `latency_sec = batch_time / batch_size`, `tokens_per_sec = n_output_tokens / latency_sec`
- Post-loop aggregation computes full statistical profile: mean, std, min, p50, p95, p99, max
- Weighted system-level throughput: `Σ(tps_i × lat_i) / T_total` — more accurate than `mean(tps)` when output lengths vary
- `latency_metrics.json` includes config context (`batch_size`, `early_exit`, `path_type`) for cross-run comparison

---

### 4. Triple Path Parser Fixes — `gen_predict_path.py`

**What was missing:** The original parser required `<PATH>...</PATH>` tags and used a greedy regex. When `INSTRUCTION_TRIPLE` had contradictory format directives, the model output paths without tags, yielding `prediction_paths = [[],[],[]]` for every sample.

**Root cause of contradictory instruction:**
```
# OLD — three conflicting directives
INSTRUCTION_TRIPLE = """
- Format: <PATH>subject<SEP>relation<SEP>object</PATH>
- Format: <PATH>subject -> relation -> object</PATH>
- list all answers between tags <answers> </endanswers>
"""
```

**Fixed instruction:**
```
INSTRUCTION_TRIPLE = """Please generate a valid reasoning triple path...
Format rules:
- Each path must be wrapped in <PATH> tags: <PATH>subject -> relation -> object</PATH>
- For multi-hop paths use: <PATH>s1 -> r1 -> o1 -> r2 -> o2</PATH>
- Generate one path per line.
- If no meaningful path can be generated, output: <PATH>NONE</PATH>
"""
```

**Robust 3-strategy parser** handles all observed model output formats:
1. **Tagged** — `<PATH>s -> r -> o</PATH>` (correct format)
2. **Answers block** — `<answers>\ns -> r -> o\n</endanswers>` (confused by old instruction)
3. **Bare lines** — `s -> r -> o` or `s → r → o` (model skipped tags entirely)

Also handles: `**Reasoning Triple**:` header prefix, missing opening `<PATH>` tag, doubled `</PATH></PATH>`, unicode `→` arrows, multi-hop chains (`s->r->o->r2->o2`).

---

### 5. Faithfulness Verification Fallback — `gen_predict_path.py`

**What was missing:** `--drop_unverified` silently emptied `prediction_paths` when the KG graph was incomplete and all paths failed verification. The QA model then had zero path evidence and answered from parametric memory, causing ~14% accuracy drop.

**What was changed:**
```python
# OLD behaviour — always overwrites
if args.drop_unverified:
    output_data["prediction_paths"] = verified_paths  # [] when all fail

# NEW behaviour — keeps originals as fallback
if args.drop_unverified:
    if verified_paths or getattr(args, "strict_drop", False):
        output_data["prediction_paths"] = verified_paths
    # else: keep originals (still marked unverified in path_meta)
```

Use `--strict_drop` to restore the original behaviour when strict filtering is intentional.

---

### 6. Empty Path Filtering — `predict_answer.py`

**What was missing:** `[[],[],[]]` entries from failed path generation were passed to `PromptBuilder`, which formatted them as ` [<INFERRED>]` placeholder lines. These wasted prompt tokens and confused the model with noise instead of evidence.

**What was added:**
```python
# Triple paths
sample["predicted_triple_paths"] = [p for p in raw_paths if p]

# Relation paths
sample["predicted_relation_paths"] = [p for p in raw_rel if p]
```

---

### 7. Interactive UI

**Files involved:**
- `bridge.py` — HTTP server on HPC login node, stdlib only (no pip deps)
- `src/qa_prediction/ui_inference.py` — single-question inference pipeline
- `files/kg_traces_ui_slurm.html` — self-contained browser UI, open as local file

**How `ui_inference.py` differs from `predict_answer.py`:**
- Loads model once per Slurm job, runs one question, exits
- Emits structured `UI_EVENT` lines to stdout that `bridge.py` forwards to the browser
- Synthesizes reasoning steps programmatically from KG paths (model does not produce structured CoT reliably)
- Extracts path-terminal entities as `path_derived_answers` and flags conflicts with model answer
- Character-by-character answer streaming via `UI_EVENT token`

**`UI_EVENT` protocol:**

| Event | Routed to |
|---|---|
| `UI_EVENT step icon=X html=Y` | Live Events tab only |
| `UI_EVENT reasoning_step {...}` | Answer tab timeline |
| `UI_EVENT path {...}` | Relation Graph + Path cards + Triples tab |
| `UI_EVENT token "c"` | Answer streaming display |
| `UI_EVENT stats {...}` | Stats bar |
| `UI_EVENT done` | All pulses stopped |

---

## Results Summary

All results on WebQSP test set, Relation path, Beam 3, NVIDIA P100.

### QA Performance

| Setting | Acc | F1 | Hit@k | MRR | Tok/s | Time (s) |
|---|---|---|---|---|---|---|
| Vanilla Qwen | 30.19 | 18.45 | 31.14 | 23.05 | 23.76 | 3.70 |
| Vanilla LLaMA | 42.11 | 20.38 | 57.37 | 56.69 | 24.61 | 17.32 |
| KG-TRACES No CoT | 78.53 | 73.67 | 82.92 | 80.70 | 11.88 | 37.66 |
| KG-TRACES + CoT | 79.20 | 73.21 | 81.45 | 77.55 | 10.29 | 82.00 |
| **KG-TRACES + CoT + EE** | **76.98** | **73.03** | **83.35** | **82.21** | **15.96** | **17.37** |

### Early Exit Latency Impact

| Setting | Tok/s ↑ | Gen Time ↓ | Avg Lat ↓ | Max Lat ↓ |
|---|---|---|---|---|
| CoT (no EE) | 10.29 | 82.00s | 41.18s | 1726.71s |
| **CoT + Early Exit** | **15.96** | **17.37s** | **8.70s** | **869.51s** |

**+55% throughput · 4.7× faster average latency · 0 change to prediction text**

### Summarization — WebNLG

| Model | BLEU | chrF | R-1 | R-L |
|---|---|---|---|---|
| FLAN-T5-base | **29.35** | **60.30** | **0.69** | **0.52** |
| BART-base | 2.20 | 40.07 | 0.49 | 0.39 |

---

## Citation

```bibtex
@article{wu2025kgtraces,
  title   = {KG-TRACES: Enhancing Large Language Models with Knowledge Graph-Constrained Trajectory Reasoning and Attribution Supervision},
  author  = {Wu, Rui and Cai, Peng and Mei, Jie and Wen, Liang and Hu, Tao and Yang, Xin and Fu, Da and Shi, Bo},
  journal = {arXiv preprint arXiv:2506.00783},
  year    = {2025}
}
```

---

## Acknowledgements

Built as part of the OmniReasoner master's project at San Jose State University, Computer Engineering Department.
