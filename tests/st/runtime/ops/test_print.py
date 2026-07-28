# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime coverage for device-side tile printing."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 1
N = 8


@pl.program
class PrintProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        src: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile = pl.load(src, [0, 0], [M, N])
        pl.tile.print(tile)
        return pl.store(tile, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        return self.kernel(src, out)


class PrintCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform: str):
        super().__init__(platform=platform)

    def get_name(self) -> str:
        return "print_fp32_values"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [M, N],
                DataType.FP32,
                init_value=lambda: torch.arange(N).reshape(M, N).float(),
            ),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return PrintProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["src"]


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
def test_print(test_runner, platform):
    result = test_runner.run(PrintCase(platform=platform))
    assert result.passed, f"Test failed: {result.error}"
