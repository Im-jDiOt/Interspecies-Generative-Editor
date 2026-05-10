"""
Align raw face photos to 512x512 ArcFace template, save as PNG.

Usage:
    python scripts/prepare_faces.py --in data/faces/raw --out data/faces/aligned

Each output file is named after the input stem: <stem>.png
Photos that fail detection/alignment are reported and skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.faces import align_face  # noqa: E402
from PIL import Image  # noqa: E402


SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    inputs = sorted(p for p in src.iterdir() if p.suffix.lower() in SUFFIXES)
    if not inputs:
        print(f"no images found in {src}")
        return 1

    failed = []
    for p in inputs:
        try:
            face = align_face(Image.open(p), size=args.size)
        except Exception as e:
            failed.append((p.name, str(e)))
            print(f"  [skip] {p.name}: {e}")
            continue
        out_path = dst / f"{p.stem}.png"
        face.image.save(out_path)
        # embedding을 정렬 시점에 저장 — 추론 때 재감지 불필요
        import numpy as np
        np.save(out_path.with_suffix(".npy"), face.embedding)
        print(f"  [ok]   {p.name}  ->  {out_path.name}  (det_score={face.bbox_score:.2f})")

    print(f"\ndone: {len(inputs) - len(failed)}/{len(inputs)} aligned")
    if failed:
        print("failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
