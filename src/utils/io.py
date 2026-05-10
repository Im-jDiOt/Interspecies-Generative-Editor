from pathlib import Path

import torch


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def ensure_dir(p) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path
