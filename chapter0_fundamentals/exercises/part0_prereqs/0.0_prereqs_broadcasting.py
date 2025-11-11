import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

x = t.ones((3, 1, 5))
print(x.unsqueeze(2).shape)
print(x.shape)
print(x.squeeze(1).shape)
print(x.squeeze(0).shape)