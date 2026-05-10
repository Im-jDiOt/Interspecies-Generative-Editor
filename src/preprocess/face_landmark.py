"""
MediaPipe FaceMesh -> control image expected by CrucibleAI/ControlNetMediaPipeFace.

The trained ControlNet expects a black-background image with the tesselation,
contours, and iris connections drawn in MediaPipe's default colored styles.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )


def render_facemesh_control(pil: Image.Image) -> Optional[Image.Image]:
    """
    Run FaceMesh on `pil` (RGB) and render the canonical control image.
    Returns None if no face is detected.
    """
    import mediapipe as mp
    from mediapipe.python.solutions import drawing_styles, drawing_utils, face_mesh

    img = np.array(pil.convert("RGB"))
    h, w = img.shape[:2]

    res = _facemesh().process(img)
    if not res.multi_face_landmarks:
        return None

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    landmarks = res.multi_face_landmarks[0]

    drawing_utils.draw_landmarks(
        image=canvas,
        landmark_list=landmarks,
        connections=face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    drawing_utils.draw_landmarks(
        image=canvas,
        landmark_list=landmarks,
        connections=face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )
    drawing_utils.draw_landmarks(
        image=canvas,
        landmark_list=landmarks,
        connections=face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
    )
    return Image.fromarray(canvas)


def render_openpose_control(pil: Image.Image) -> Optional[Image.Image]:
    """Fallback path for the OpenPose ControlNet — uses controlnet_aux."""
    from controlnet_aux import OpenposeDetector
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return detector(pil)


def landmark_cache_path(face_path: Path, kind: str) -> Path:
    return face_path.with_name(f"{face_path.stem}__control_{kind}.png")


def get_or_make_control(face_path: Path, kind: str = "facemesh") -> Image.Image:
    cache = landmark_cache_path(face_path, kind)
    if cache.exists():
        return Image.open(cache).convert("RGB")
    pil = Image.open(face_path).convert("RGB")
    if kind == "facemesh":
        ctrl = render_facemesh_control(pil)
    elif kind == "openpose":
        ctrl = render_openpose_control(pil)
    else:
        raise ValueError(f"unknown control kind: {kind}")
    if ctrl is None:
        raise RuntimeError(f"control extraction ({kind}) failed for {face_path}")
    ctrl.save(cache)
    return ctrl
