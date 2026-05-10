import random

import numpy as np
import torch


def make_generator(seed: int, device: str) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
