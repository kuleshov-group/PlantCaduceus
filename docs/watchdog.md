# Training Watchdog for Expression Fine-Tuning

**Status:** v1 (working, tested) — further improvements possible (8/25/2026)
**Wraps:** `src/run_finetune_expression.sh`

## Purpose

Our LoRA fine-tuning runs on the 2080 Ti hardware occasionally hit a known instability: the Mamba2/Triton kernel combination can produce a NaN/Inf loss partway through training (documented in the training script itself, roughly a 40% failure rate per attempt). When this happens, the run needs to be killed and restarted — previously a manual, watch-and-babysit process.

`watchdog.sh` automates this. It launches training, monitors the log for NaN/Inf, kills and restarts automatically if it appears, and gives up after a configurable number of attempts. This follows the monitoring approach suggested in the original project notes for exactly this failure mode.

## What it does

1. Launches `run_finetune_expression.sh` inside the Docker container.
2. Polls the training log at a set interval, checking for NaN/Inf in loss or gradient norm.
3. If detected: kills the run immediately (before a bad checkpoint can be saved) and discards any checkpoints from that attempt.
4. If the run crashes for another reason (e.g. out-of-memory) and a checkpoint exists: resumes training from the last saved checkpoint instead of starting over.
5. Retries with a short backoff, up to a configurable maximum, then stops and reports if the failures don't resolve.

## How to use it

Run from the repo root, same environment variables as the underlying training script:

```bash
DATA_PREFIX=./data/expression_dataset_1024 \
MODEL_PATH=./model/PlantCAD2-Small-l24-d0768 \
NUM_EPOCHS=5 \
MAX_RETRIES=5 \
./watchdog.sh
```

Key options (all optional, sensible defaults built in):

| Variable | Purpose | Default |
|---|---|---|
| `MAX_RETRIES` | Give up after this many failed attempts | 5 |
| `CHECK_INTERVAL` | Seconds between log checks | 30 |
| `SANITY` | Set to `1` for a fast smoke test before a real run | 0 |
| `CUDA_VISIBLE_DEVICES` | Which GPU to use (useful on the shared server) | 0 |

## Current limitations / next steps

This is a first version, built and tested over the past couple of days of full runs. A few things worth improving:

- **Detection speed is polling-based**, so there's a short delay (up to one `CHECK_INTERVAL`) between a NaN appearing and the run being killed. Fine for multi-hour runs; could be tightened further if needed.
- **GPU selection is manual.** On our shared server, other users' jobs can occupy GPU 0; the watchdog now supports pointing at a specific GPU, but doesn't yet check availability automatically.
- **No automated summary report** at the end of a multi-attempt run (currently: check the logs by hand). Would be a nice addition once we're running this more routinely.

## Result so far

Using this, I've been running expression fine-tuning attempts on PlantCAD2-Small, with the watchdog recovering automatically from NaN'd attempts along the way without needing to babysit each one.
