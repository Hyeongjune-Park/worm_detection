# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Human-in-the-loop larva/insect tracking system for laboratory video footage (up to ~10 insects). Uses a per-track ROI-based, multi-sensor cross-validated pipeline with Kalman filtering. User provides initial seed boxes and arena region; the system tracks center/head coordinates per frame using SAM2 + Template matching + KLT optical flow sensors with QA fusion.

## Setup & Commands

```bash
# Environment setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install numpy opencv-python pyyaml scipy
# SAM2 optional: pip install torch torchvision (+ sam2 submodule)

# 1) Arena selection
python select_arena.py --input video.mp4 --out arena.yaml

# 2) Seed selection (initial bbox per larva)
python select_seeds.py --input video.mp4 --arena arena.yaml --out seeds.yaml

# 3) Run tracking pipeline
python run.py --config config.yaml --input video.mp4 --output output/run1 --seeds seeds.yaml --arena arena.yaml
```

No test framework, linter, or CI/CD is configured.

## Architecture

**Per-track ROI loop (per frame, per track):**
```
Kalman predict → ROI crop → 3 Sensors (SAM2, Template, KLT) → QA Fusion → KF update → State machine → Head estimation → Output
```

### Core Modules

- **`run.py`** — Entry point; loads config/arena/seeds, invokes pipeline
- **`pipeline/runner.py`** — Main orchestration: creates tracks from seeds, runs per-track sensor loop each frame
- **`sensors/`** — Three sensor implementations + head estimator:
  - `base.py` — `Sensor` ABC + `SensorResult` dataclass
  - `sam2_sensor.py` — SAM2 image predictor → mask → centroid/PCA endpoints/border_touch. Falls back to `MotionMaskSensor`
  - `template_sensor.py` — Gray+Sobel edge NCC matching in ROI. Slow-updates template on high quality
  - `klt_sensor.py` — KLT optical flow with forward-backward error filtering. Median of valid points = center
  - `head_estimator.py` — PCA endpoints + movement direction alignment → head point
- **`qa/fusion.py`** — Cross-validation: computes d_pred/d_tpl/d_klt distances + border_touch. Three cases: A (accept SAM2), B (fallback to TPL/KLT), C (predict-only)
- **`tracking/`** — Track data model, Kalman filter, state machine:
  - `track.py` — `Track` dataclass (7-state enum, sensor memory, quality history)
  - `kalman.py` — `KalmanFilter2D` [x, y, vx, vy] with configurable Q/R noise
  - `state_machine.py` — ACTIVE↔UNCERTAIN↔OCCLUDED→NEEDS_RESEED, MERGED, EXITED, DEAD_CANDIDATE
- **`roi/roi_manager.py`** — Fixed-size ROI centered on Kalman prediction, arena-clamped
- **`preprocess/`** — Grayscale, denoise, optional CLAHE
- **`io_utils/`** — VideoReader, OverlayWriter, ArtifactWriter (CSV), EventWriter (JSONL), overlay drawing

### Legacy Modules (not used by current pipeline, kept for reference)

- `detection/` — Background subtraction + blob extraction (was used in auto-detection pipeline)
- `tracking/multi_tracker.py`, `tracking/association.py` — Hungarian assignment-based MOT (replaced by seed-based per-track loop)
- `segmentation/` — Old `Segmenter` ABC (replaced by `sensors/`)

## Key Design Rules (from design_spec.md)

- **SAM2 output is NOT ground truth** — only accepted after QA cross-validation
- **No bbox/area-based rollback** — judge by coordinate consistency across sensors
- **No infinite ROI expansion** — failures accumulate → NEEDS_RESEED
- **No track merging on overlap** — use MERGED state, keep separate tracks
- **Goal is center/head coordinates**, not mask quality
- **Human seed required** — no auto-detection of new tracks

## Output

- `tracks.csv` — Per-frame: frame_idx, time_sec, track_id, center_x/y, head_x/y, state, quality_score, sensor_used, bbox, area
- `events.jsonl` — State transitions, merge events
- `overlay.mp4` — Annotated video (bbox, center, head arrow, state color)
