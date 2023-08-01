import random
import numpy as np
import torch as th
import torch.nn as nn
from typing import List


def get_parameters(*models: nn.Module) -> List[nn.Parameter]:
    params = []
    for model in models:
        params.extend(model.parameters())
    
    return params

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)