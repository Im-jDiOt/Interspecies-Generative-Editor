"""
Single entry point for both Stage 2 and Stage 3 sweeps.

Usage:
    python scripts/run_sweep.py --config configs/stage2_ipadapter.yaml
    python scripts/run_sweep.py --config configs/stage3_pose.yaml --limit 1
    python scripts/run_sweep.py --config <cfg> --dry-run

The config's `pipeline` key picks the pipeline class. Each generation produces
one PNG and one row in metadata.parquet, indexed by all sweep dimensions.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.utils.io import ensure_dir  # noqa: E402


BASE_CFG = Path("configs/base.yaml")


def load_cfg(path: Path):
    base = OmegaConf.load(BASE_CFG)
    stage = OmegaConf.load(path)
    return OmegaConf.merge(base, stage)


def expand_grid(sweep) -> list[dict]:
    keys = list(sweep.keys())
    values = [list(sweep[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def make_run_dir(cfg) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg.output_dir).parent / f"stage{cfg.stage}_{ts}"
    ensure_dir(out / "images")
    OmegaConf.save(cfg, out / "config.yaml")
    return out


def build_pipeline(cfg):
    name = cfg.pipeline
    if name == "stage2_ipadapter":
        from src.pipelines.stage2_ipadapter import Stage2Pipeline
        return Stage2Pipeline(cfg)
    if name == "stage3_pose_aware":
        from src.pipelines.stage3_pose_aware import Stage3Pipeline
        return Stage3Pipeline(cfg)
    raise ValueError(f"unknown pipeline: {name}")


def filename_for(face_path: Path, combo: dict) -> str:
    parts = [face_path.stem, combo["species"]]
    parts.append(f"ip{combo['ip_scale']:.2f}")
    if "controlnet_scale" in combo:
        parts.append(f"cn{combo['controlnet_scale']:.2f}")
    parts.append(f"seed{combo['seed']}")
    return "__".join(parts) + ".png"


def call_pipeline(pipeline, face_path: Path, combo: dict):
    if "controlnet_scale" in combo:
        return pipeline.generate(
            face_path=face_path,
            species_key=combo["species"],
            ip_scale=combo["ip_scale"],
            controlnet_scale=combo["controlnet_scale"],
            seed=combo["seed"],
        )
    return pipeline.generate(
        face_path=face_path,
        species_key=combo["species"],
        ip_scale=combo["ip_scale"],
        seed=combo["seed"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N (face, combo) pairs (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    faces = sorted(Path().glob(cfg.faces_glob))
    if not faces:
        print(f"no faces match {cfg.faces_glob}")
        return 1

    grid = expand_grid(cfg.sweep)
    pairs = [(f, c) for f in faces for c in grid]
    print(f"faces={len(faces)}  combos={len(grid)}  total={len(pairs)}")

    if args.limit is not None:
        pairs = pairs[: args.limit]
        print(f"--limit {args.limit} -> running {len(pairs)}")

    if args.dry_run:
        for f, c in pairs[:10]:
            print(" ", f.name, c)
        if len(pairs) > 10:
            print(f"  ... (+{len(pairs) - 10} more)")
        return 0

    run_dir = make_run_dir(cfg)
    pipeline = build_pipeline(cfg)

    rows = []
    img_dir = run_dir / "images"
    t0 = time.time()
    for i, (face_path, combo) in enumerate(pairs, 1):
        try:
            out = call_pipeline(pipeline, face_path, combo)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{i}/{len(pairs)}] FAIL {face_path.name} {combo}: {e}")
            rows.append({
                "face_id": face_path.stem,
                "face_path": str(face_path),
                "image_path": "",
                "stage": int(cfg.stage),
                "error": str(e),
                "notes": "",
                **combo,
            })
            continue

        fname = filename_for(face_path, combo)
        img_path = img_dir / fname
        out.image.save(img_path)
        rows.append({
            "face_id": face_path.stem,
            "face_path": str(face_path),
            "image_path": str(img_path),
            "stage": int(cfg.stage),
            "error": "",
            "notes": "",
            **combo,
        })
        elapsed = time.time() - t0
        avg = elapsed / i
        eta = avg * (len(pairs) - i)
        print(f"[{i}/{len(pairs)}] {fname}  ({avg:.1f}s/it, eta {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "metadata.parquet", index=False)
    print(f"\nrun_dir: {run_dir}")
    print(f"images:  {img_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
