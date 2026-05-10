"""
Pre-download all model weights so the first sweep run doesn't stall on huggingface.

Usage:
    python scripts/setup_models.py

Skips items that fail individually so a partial setup is still useful.
"""
from __future__ import annotations

import sys
import traceback


REPOS = [
    ("StableDiffusionPipeline", "runwayml/stable-diffusion-v1-5"),
    ("IP-Adapter-FaceID weights", "h94/IP-Adapter-FaceID"),
    ("CLIP-ViT-H (image encoder for FaceID-Plus, optional)",
     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    # ("ControlNet MediaPipe Face", "CrucibleAI/ControlNetMediaPipeFace"),
    # ("ControlNet OpenPose (fallback)", "lllyasviel/control_v11p_sd15_openpose"),
]


def fetch(repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, resume_download=True)


def main() -> int:
    failures = []
    for label, repo_id in REPOS:
        print(f"[fetch] {label}: {repo_id}")
        try:
            fetch(repo_id)
        except Exception:
            traceback.print_exc()
            failures.append(repo_id)

    print("\n[setup_models] insightface buffalo_l will be downloaded on first FaceAnalysis() call.")

    if failures:
        print("\n[setup_models] Some repos failed:")
        for r in failures:
            print(f"  - {r}")
        return 1
    print("\n[setup_models] all good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
