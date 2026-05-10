"""
Face detection + similarity-transform alignment to a 512x512 canvas.

We rely on insightface (buffalo_l) for both 5-keypoint detection and ArcFace
embedding so the same model that drives IP-Adapter-FaceID conditioning also
defines the alignment frame. That way the embedding the IP-Adapter sees comes
from a well-aligned face every time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# Standard ArcFace 5-point template at 112x112, scaled to 512x512.
_ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _dst_template(size: int) -> np.ndarray:
    return _ARCFACE_DST_112 * (size / 112.0)


@dataclass
class AlignedFace:
    image: Image.Image           # RGB, size x size
    embedding: np.ndarray        # (512,) ArcFace, L2-normalized
    landmarks_512: np.ndarray    # (5, 2) keypoints in the aligned frame
    bbox_score: float


@lru_cache(maxsize=1)
def _face_app(det_size: int = 640):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def _largest_face(faces):
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(faces, key=area)


def align_face(pil: Image.Image, size: int = 512) -> AlignedFace:
    """
    Detect the largest face in `pil`, align it to `size`x`size` via similarity
    transform onto the ArcFace template, and return the aligned image plus
    its ArcFace embedding.
    """
    rgb = np.array(pil.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()

    app = _face_app()
    faces = app.get(bgr)
    if not faces:
        raise RuntimeError("no face detected")

    f = _largest_face(faces)
    src = f.kps.astype(np.float32)            # (5, 2) in original image
    dst = _dst_template(size)

    # Estimate similarity (rotation + uniform scale + translation).
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("alignment failed")

    aligned_bgr = cv2.warpAffine(bgr, M, (size, size), borderValue=0)
    aligned_rgb = aligned_bgr[:, :, ::-1].copy()

    # Push original 5-keypoints through M to get coordinates in aligned frame.
    src_h = np.concatenate([src, np.ones((5, 1), dtype=np.float32)], axis=1)
    aligned_kps = src_h @ M.T

    return AlignedFace(
        image=Image.fromarray(aligned_rgb),
        embedding=f.normed_embedding.astype(np.float32),
        landmarks_512=aligned_kps.astype(np.float32),
        bbox_score=float(f.det_score),
    )


def load_aligned(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def arcface_embedding(pil: Image.Image) -> Optional[np.ndarray]:
    """Re-extract ArcFace embedding from an already-aligned image."""
    rgb = np.array(pil.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    faces = _face_app().get(bgr)
    if not faces:
        return None
    return _largest_face(faces).normed_embedding.astype(np.float32)
