# run.py — Human-in-the-Loop Multi-Sensor Tracking Pipeline
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import yaml

from pipeline.runner import run_pipeline


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: dict, b: dict) -> dict:
    """merge b into a (recursive), return new dict"""
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def inject_arena(cfg: dict, arena_yaml: dict) -> dict:
    """arena_manual.yaml → config arena section에 주입."""
    am = (arena_yaml or {}).get("arena_manual", None)
    if not isinstance(am, dict):
        return cfg

    rect = {
        "enabled": True,
        "x0": int(am["x0"]),
        "y0": int(am["y0"]),
        "x1": int(am["x1"]),
        "y1": int(am["y1"]),
    }
    patch = {
        "arena": {
            "enabled": True,
            "manual_rect": rect,
        }
    }
    return deep_merge(cfg, patch)


def main():
    ap = argparse.ArgumentParser(description="Insect tracking pipeline (human-in-the-loop)")
    ap.add_argument("--config", default="config.yaml", help="config yaml")
    ap.add_argument("--input", required=True, help="video file or frames dir")
    ap.add_argument("--output", default="", help="output dir (default: output/YYYYMMDD_HHMMSS)")
    ap.add_argument("--seeds", required=True, help="seeds yaml from select_seeds.py")
    ap.add_argument("--arena", default="", help="arena yaml from select_arena.py (optional)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # output 자동 생성
    output_dir = args.output
    if not output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/{ts}"

    if args.arena:
        arena_cfg = load_yaml(args.arena)
        cfg = inject_arena(cfg, arena_cfg)

    # input path override
    cfg.setdefault("input", {})["path"] = args.input

    # output dir override
    cfg.setdefault("project", {})["output_dir"] = output_dir

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        cfg=cfg,
        input_path=args.input,
        output_dir=str(out_dir),
        seeds_path=args.seeds,
    )


if __name__ == "__main__":
    main()
