# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from ....utils import cute_op


@cute_op(f"{LIBRARY_NAME}::continuous_count_cuda", mutates_args={"output", "sorted_output", "sorted_indices"})
@cpp_jit()
def continuous_count_cuda(
    x: torch.Tensor,
    output: torch.Tensor,
    sorted_output: torch.Tensor | None,
    sorted_indices: torch.Tensor | None,
    E: int,
    BLOCK_SIZE: int,
) -> None: ...
