import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seeds everything so that experiments are deterministic.

    Args:
        seed (int): Seed value.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
