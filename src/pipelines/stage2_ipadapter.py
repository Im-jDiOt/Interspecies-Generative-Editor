"""
Stage 2 — IP-Adapter-plus-face conditioning.

얼굴 이미지를 CLIP image encoder로 인코딩하여 IP-Adapter cross-attention에 주입.
ArcFace 기반 FaceID 대비 버전 호환성이 높고 안정적.
identity 보존력은 약간 낮지만 파이프라인 동작이 우선.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class Stage2Output:
    image: Image.Image
    ip_scale: float
    species: str


class Stage2Pipeline:
    def __init__(self, cfg: DictConfig):
        from diffusers import StableDiffusionPipeline

        self.cfg = cfg
        self.device, self.dtype = resolve_runtime(cfg)

        self.pipe = StableDiffusionPipeline.from_pretrained(**load_sd_kwargs(cfg))
        ip = cfg.ip_adapter
        self.pipe.load_ip_adapter(
            ip.repo,
            subfolder=ip.get("subfolder", None),
            weight_name=ip.weight,
            image_encoder_folder=ip.get("image_encoder", None),
        )
        to_device(self.pipe, cfg)
        apply_memory_tweaks(self.pipe, cfg)

        self._face_cache: dict[str, Image.Image] = {}

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
        seed: int,
    ) -> Stage2Output:
        prompt, neg = build_prompts(self.cfg, species_key)
        face_img = self._load_face(face_path)

        self.pipe.set_ip_adapter_scale(float(ip_scale))
        gen_device = self.device if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=gen_device).manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            ip_adapter_image=face_img,
            num_inference_steps=int(self.cfg.num_inference_steps),
            guidance_scale=float(self.cfg.guidance_scale),
            height=int(self.cfg.height),
            width=int(self.cfg.width),
            generator=gen,
        ).images[0]

        return Stage2Output(image=out, ip_scale=float(ip_scale), species=species_key)
