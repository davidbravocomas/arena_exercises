import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

from utils import display_array_as_img, display_soln_array_as_img

arr = np.load("numbers.npy")

# arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
# display_array_as_img(arr1)

# arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
# display_array_as_img(arr2)

# arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
# display_array_as_img(arr3)

arr5 = einops.repeat(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
