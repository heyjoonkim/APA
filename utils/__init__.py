
import os
import random

import numpy as np
import torch

def seed_everything(seed: int) -> None:
    print(f'Initialize random seed to {seed}')
    assert seed is not None, 'Random seed cannot be None.'
   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False