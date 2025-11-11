import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

arr = np.load(section_dir / "numbers.npy")

# print(arr[0].shape)
# display_array_as_img(arr[0])  # plotting the first image in the batch

# print(arr[0, 0].shape)
# display_array_as_img(arr[0, 0])  # plotting the first channel of the first image, as monochrome

# Exercise 1
# arr_stacked = einops.rearrange(arr, "b c h w -> c (b h) w")
# print(arr_stacked.shape)
# display_array_as_img(arr_stacked)  # plotting all images, stacked in a column

# Exercise 2
# arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
# display_array_as_img(arr2)

# Exercise 3
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)