"""
Common pipeline plumbing. Subclasses pick which diffusers pipeline class to
load and which conditioners to attach, but share the same memory tweaks and
the same ArcFace-embedding-prep contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.io import resolve_dtype


@dataclass
class GenContext:
    prompt: str
    negative_prompt: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int


def apply_memory_tweaks(pipe, cfg: DictConfig) -> None:
    mem = cfg.get("memory", {}) or {}
    if mem.get("attention_slicing", True):
        pipe.enable_attention_slicing()
    if mem.get("vae_slicing", True):
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass
    if mem.get("cpu_offload", False):
        pipe.enable_model_cpu_offload()


def to_device(pipe, cfg: DictConfig) -> None:
    if not (cfg.get("memory", {}) or {}).get("cpu_offload", False):
        pipe.to(cfg.device)


def make_id_embeds(
    arcface_512: np.ndarray,
    device: str,
    dtype: torch.dtype,
    do_cfg: bool = True,
) -> torch.Tensor:
    """
    Format ArcFace embedding for diffusers' IP-Adapter-FaceID.
    diffusers >= 0.27 는 CFG를 내부에서 처리하므로 (1, 1, 512)만 넘기면 된다.
    구버전 호환용으로 do_cfg=True면 (2, 1, 512)로 concat하지만,
    'tuple has no shape' 에러 시 do_cfg=False로 호출할 것.
    """
    emb = torch.from_numpy(arcface_512.astype(np.float32)).reshape(1, 1, -1)
    emb = emb.to(device=device, dtype=dtype)
    if do_cfg:
        neg = torch.zeros_like(emb)
        emb = torch.cat([neg, emb], dim=0)  # (2, 1, 512)
    return emb


def resolve_runtime(cfg: DictConfig) -> tuple[str, torch.dtype]:
    return cfg.device, resolve_dtype(cfg.dtype)


def load_sd_kwargs(cfg: DictConfig) -> dict:
    return dict(
        pretrained_model_name_or_path=cfg.sd_path,
        torch_dtype=resolve_dtype(cfg.dtype),
        safety_checker=None,
        requires_safety_checker=False,
    )


def species_text(cfg: DictConfig, species_key: str) -> str:
    mapping: Optional[dict] = cfg.get("species_text", None)
    if mapping is None:
        return species_key
    return mapping.get(species_key, species_key)


def build_prompts(cfg: DictConfig, species_key: str) -> tuple[str, str]:
    text = species_text(cfg, species_key)
    return cfg.prompt_template.format(species_text=text), cfg.negative_prompt
