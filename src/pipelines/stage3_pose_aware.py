"""
Stage 3 — IP-Adapter-FaceID + ControlNet (face landmark).

Identity comes from ArcFace via IP-Adapter; spatial structure (head pose,
facial layout) is enforced by ControlNet on a precomputed FaceMesh image.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from omegaconf import DictConfig

from src.pipelines.base import (
    apply_memory_tweaks,
    build_prompts,
    load_sd_kwargs,
    resolve_runtime,
    to_device,
)
from src.preprocess.face_landmark import get_or_make_control


@dataclass
class Stage3Output:
    image: Image.Image
    ip_scale: float
    controlnet_scale: float
    species: str


def _load_controlnet(cfg: DictConfig, dtype):
    from diffusers import ControlNetModel
    cn = cfg.controlnet
    try:
        return ControlNetModel.from_pretrained(
            cn.repo,
            subfolder=cn.get("subfolder", None),
            torch_dtype=dtype,
        )
    except Exception as e:
        fb = cn.get("fallback_repo", None)
        if not fb:
            raise
        print(f"[stage3] primary ControlNet failed ({e}); falling back to {fb}")
        return ControlNetModel.from_pretrained(fb, torch_dtype=dtype)


class Stage3Pipeline:
    def __init__(self, cfg: DictConfig):
        from diffusers import StableDiffusionControlNetPipeline

        self.cfg = cfg
        self.device, self.dtype = resolve_runtime(cfg)

        controlnet = _load_controlnet(cfg, self.dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            controlnet=controlnet,
            **load_sd_kwargs(cfg),
        )

        ip = cfg.ip_adapter
        self.pipe.load_ip_adapter(
            ip.repo,
            subfolder=None,
            weight_name=ip.weight,
            image_encoder_folder=ip.get("image_encoder", None),
        )

        to_device(self.pipe, cfg)
        apply_memory_tweaks(self.pipe, cfg)

        self._face_cache: dict[str, Image.Image] = {}
        self.control_kind = cfg.controlnet.get("control_kind", "facemesh")

    @lru_cache(maxsize=64)
    def _control_image(self, face_path_str: str) -> Image.Image:
        return get_or_make_control(Path(face_path_str), kind=self.control_kind)

    def _load_face(self, face_path: Path) -> Image.Image:
        key = str(face_path)
        if key not in self._face_cache:
            self._face_cache[key] = Image.open(face_path).convert("RGB")
        return self._face_cache[key]

    def generate(
        self,
        face_path: Path,
        species_key: str,
        ip_scale: float,
        controlnet_scale: float,
        seed: int,
    ) -> Stage3Output:
        prompt, neg = build_prompts(self.cfg, species_key)
        face_img = self._load_face(face_path)
        ctrl_img = self._control_image(str(face_path))

        self.pipe.set_ip_adapter_scale(float(ip_scale))
        gen_device = self.device if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=gen_device).manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            image=ctrl_img,
            ip_adapter_image=face_img,
            controlnet_conditioning_scale=float(controlnet_scale),
            num_inference_steps=int(self.cfg.num_inference_steps),
            guidance_scale=float(self.cfg.guidance_scale),
            height=int(self.cfg.height),
            width=int(self.cfg.width),
            generator=gen,
        ).images[0]

        return Stage3Output(
            image=out,
            ip_scale=float(ip_scale),
            controlnet_scale=float(controlnet_scale),
            species=species_key,
        )
