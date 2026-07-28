# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime coverage for per-row indexed TCONCAT."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 8
N = 64
IDX_COLS = 8


def _src0() -> torch.Tensor:
    return torch.arange(M * N, dtype=torch.float32).reshape(M, N).contiguous()


def _src1() -> torch.Tensor:
    return (1000 + torch.arange(M * N, dtype=torch.float32)).reshape(M, N).contiguous()


def _idx0() -> torch.Tensor:
    result = torch.zeros((M, IDX_COLS), dtype=torch.int32)
    result[:, 0] = torch.tensor([0, 1, 7, 16, 31, 32, 48, 64], dtype=torch.int32)
    return result


def _idx1() -> torch.Tensor:
    result = torch.zeros((M, IDX_COLS), dtype=torch.int32)
    result[:, 0] = torch.tensor([64, 40, 20, 16, 8, 32, 30, 0], dtype=torch.int32)
    return result


@pl.program
class ConcatIdxProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        src0: pl.Tensor[[M, N], pl.FP32],
        src1: pl.Tensor[[M, N], pl.FP32],
        idx0: pl.Tensor[[M, IDX_COLS], pl.INT32],
        idx1: pl.Tensor[[M, IDX_COLS], pl.INT32],
        out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        value0 = pl.load(src0, [0, 0], [M, N])
        value1 = pl.load(src1, [0, 0], [M, N])
        count0 = pl.load(idx0, [0, 0], [M, IDX_COLS], valid_shapes=[M, 1])
        count1 = pl.load(idx1, [0, 0], [M, IDX_COLS], valid_shapes=[M, 1])
        dst = pl.load(out, [0, 0], [M, N])
        result = pl.tile.concat_idx(value0, value1, count0, count1, dst)
        return pl.store(result, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        src0: pl.Tensor[[M, N], pl.FP32],
        src1: pl.Tensor[[M, N], pl.FP32],
        idx0: pl.Tensor[[M, IDX_COLS], pl.INT32],
        idx1: pl.Tensor[[M, IDX_COLS], pl.INT32],
        out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        return self.kernel(src0, src1, idx0, idx1, out)


class ConcatIdxCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform: str):
        super().__init__(platform=platform)

    def get_name(self) -> str:
        return "concat_idx_per_row_boundaries"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src0", [M, N], DataType.FP32, init_value=_src0),
            TensorSpec("src1", [M, N], DataType.FP32, init_value=_src1),
            TensorSpec("idx0", [M, IDX_COLS], DataType.INT32, init_value=_idx0),
            TensorSpec("idx1", [M, IDX_COLS], DataType.INT32, init_value=_idx1),
            TensorSpec("out", [M, N], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        return ConcatIdxProgram

    def compute_expected(self, tensors, params=None):
        expected = torch.zeros_like(tensors["out"])
        for row in range(M):
            n0 = min(int(tensors["idx0"][row, 0]), N)
            n1 = min(int(tensors["idx1"][row, 0]), N - n0)
            expected[row, :n0] = tensors["src0"][row, :n0]
            expected[row, n0 : n0 + n1] = tensors["src1"][row, :n1]
        tensors["out"][:] = expected


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
def test_concat_idx(test_runner, platform):
    result = test_runner.run(ConcatIdxCase(platform=platform))
    assert result.passed, f"Test failed: {result.error}"
