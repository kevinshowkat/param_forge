# Param Forge Batch Run (Interactive-First)

## Overview
Param Forge currently positions "Batch run" as a placeholder. This spec defines a first-class interactive Batch run that also supports a fully headless CLI. The goal is to give independent image-model experimenters a reproducible workbench for running prompt sets across a matrix of providers/models/knobs, with concurrency and budget limits, and a complete manifest of every output.

## Goals
- Interactive batch run setup is the default and fastest path to a run.
- Headless CLI is fully supported and parity-aligned with interactive actions.
- Every run produces a single manifest that indexes all receipts and images.
- Concurrency and budget limits are enforced and visible before the run.
- Runs are resumable without duplicating completed outputs.

## Non-Goals (MVP)
- No visual grid/diff for comparing outputs.
- No auto-selection or ranking of models.
- No cloud sync or dashboard.

## Primary Users
- Independent model tinkerers comparing provider/model behavior.
- Individual researchers running parameter sweeps.
- Small teams who want lightweight local audit trails.

## UX: Interactive-First Flow

### Entry Points
- Default: `param-forge batch-run` opens the Batch run builder.
- Headless: `param-forge batch-run --prompts ... --matrix ... --out ...` runs without UI.
- Parity rule: every interactive action must map to CLI flags or manifest entries.

### Batch run builder (MVP)
1. Load Prompts
   - Select `.txt` or `.csv`.
   - Show prompt count and preview first/last few prompts.
   - Option to filter (e.g., prompt range).

2. Load Matrix
   - Select `.yaml` or `.json`.
   - Show matrix blocks and expanded job count.
   - Warnings for unknown price mappings or invalid values.

3. Review Plan
   - Show: prompt count, matrix size, total jobs, total images, estimated cost.
   - Show: provider/model breakdown and any warnings.

4. Set Limits
   - Global concurrency.
   - Per-provider concurrency.
   - Budget and budget mode.

5. Confirm & Run
   - Write `run.json` as `planned` before execution.
   - Start run with live status.

### During Run
- Live status table (queued/running/success/fail/skip).
- Summary panel (elapsed, throughput, estimated cost, budget remaining).
- Pause / Resume / Stop controls.
- Stop sets `status=cancelled` and preserves outputs.

### After Run
- Summary: success/fail/skip counts, cost, manifest path.
- Actions: open output folder, open manifest, re-run with tweaks.

## CLI (Headless)

### Primary Command
```bash
param-forge batch-run --prompts prompts.txt --matrix matrix.yaml --out runs/food_glam_v1
```

### Flags (MVP)
- `--prompts <path>`: prompt file (.txt/.csv)
- `--matrix <path>`: matrix file (.yaml/.json)
- `--out <dir>`: output run directory
- `--concurrency <int>`: global concurrency (default: 3)
- `--provider-concurrency <provider=int,...>`: per-provider cap
- `--budget <usd>`: total budget limit
- `--budget-mode <estimate|strict|off>`: default `estimate`
- `--dry-run`: validate inputs, show plan and estimated cost
- `--resume`: continue a prior run based on `run.json`

## Inputs

### Prompt Set (.txt)
- One prompt per line.
- Blank lines ignored.
- Lines starting with `#` ignored.
- Prompt IDs auto-assigned by order (`p-0001`, `p-0002`, ...).

### Prompt Set (.csv)
- Required column: `prompt`
- Optional: `id`, `metadata` (JSON string)
- Extra columns preserved as prompt metadata.

### Matrix Definition (.yaml/.json)

Schema v1:
```yaml
version: 1
defaults:
  n: 1
  output_format: jpeg
  background: null

matrix:
  - provider: openai
    model: [gpt-image-1.5, gpt-image-1-mini]
    size: [1024x1024, 1024x1536]
    seed: [101, 102]
    provider_options:
      quality: [low, high]
      moderation: [low]

  - provider: gemini
    model: [gemini-2.5-flash-image]
    size: [4:5]
    provider_options:
      image_size: [2K]

include:
  - provider: flux
    model: flux-2-flex
    size: 1024x1024
    seed: 42
    provider_options:
      guidance: 5

exclude:
  - provider: openai
    size: 1024x1536
    provider_options:
      quality: low

limits:
  concurrency: 3
  provider_concurrency:
    openai: 1
    gemini: 2
  budget_usd: 10.0
  budget_mode: estimate
```

Rules:
- Each `matrix` block expands as a cartesian product of list values.
- Scalars are fixed values.
- `include` adds explicit runs.
- `exclude` removes matching runs.
- `provider_options` are passed through as-is; unknown keys allowed.

## Execution Model

### Planning
- Load prompts and matrix.
- Expand matrix blocks into job list.
- Multiply by prompt count and `n` to get total images.
- Estimate cost per job and total.
- Write `run.json` in `planned` state.

### Scheduling
- Stable ordering: prompt order -> matrix block order -> cartesian order.
- Worker pool with global and per-provider concurrency limits.
- Default retry: 1 for transient provider/network errors.

### Resume
- `--resume` or interactive “Resume run” skips `success` jobs.
- Failed and pending jobs are re-queued.

## Budget and Costing

### Pricing Source
- `docs/pricing_reference.json` used for estimates.

### Budget Modes
- `estimate`: allow run, stop scheduling once estimate would exceed budget.
- `strict`: unknown prices are skipped; budget overrun halts new jobs.
- `off`: no budget enforcement, still compute estimates when possible.

### Accounting
- `estimated_cost_usd` per job and total recorded in manifest.
- `actual_cost_usd` populated if receipts expose usage/cost data.

## Outputs

```
runs/food_glam_v1/
  run.json
  receipt-*.json
  *.jpg / *.png / *.webp
```

### Filename Collision Prevention
- Concurrency can collide on timestamps.
- Add a short job ID or higher-precision timestamp to filenames.

## Run Manifest (run.json)

Schema v1:
```json
{
  "schema_version": 1,
  "run_id": "food_glam_v1",
  "created_at": "2026-01-06T19:03:21Z",
  "started_at": "2026-01-06T19:03:25Z",
  "completed_at": null,
  "status": "running",
  "inputs": {
    "prompts_path": "prompts.txt",
    "matrix_path": "matrix.yaml",
    "prompt_count": 24,
    "matrix_blocks": 2,
    "expanded_jobs": 96,
    "planned_images": 96
  },
  "limits": {
    "concurrency": 3,
    "provider_concurrency": {"openai": 1, "gemini": 2},
    "budget_usd": 10.0,
    "budget_mode": "estimate"
  },
  "summary": {
    "succeeded": 0,
    "failed": 0,
    "skipped": 0,
    "estimated_cost_usd": 0.0,
    "actual_cost_usd": null
  },
  "jobs": [
    {
      "job_id": "j-000001",
      "prompt_id": "p-0001",
      "prompt": "A glossy food portrait of ramen...",
      "params": {
        "provider": "openai",
        "model": "gpt-image-1.5",
        "size": "1024x1024",
        "seed": 101,
        "n": 1,
        "output_format": "jpeg",
        "background": null,
        "provider_options": {"quality": "high", "moderation": "low"}
      },
      "status": "success",
      "attempts": 1,
      "started_at": "2026-01-06T19:03:30Z",
      "completed_at": "2026-01-06T19:03:42Z",
      "estimated_cost_usd": 0.133,
      "actual_cost_usd": null,
      "artifacts": {
        "image_paths": ["openai-20260106T190342Z-00.jpg"],
        "receipt_paths": ["receipt-openai-20260106T190342Z-00.json"]
      },
      "warnings": [],
      "error": null
    }
  ]
}
```

Manifest requirements:
- Written incrementally after each job (atomic update).
- Uses relative paths for portability.
- Captures all planned jobs, including skipped/failed.

## Success Metrics
- Interactive flow is the default and fastest path to a run.
- Manifest fully indexes all outputs.
- Runs are resumable without duplicating work.
- Budget limits prevent overspend.

## Open Questions
- Should Batch run be a separate screen in the existing UI or a distinct mode?
- How much in-UI editing of matrix params is needed for MVP?
- Should per-prompt overrides be supported in CSV for MVP?
