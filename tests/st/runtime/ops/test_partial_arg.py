# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime coverage for ``tpartargmax`` and ``tpartargmin``.

The second value/index pair has a smaller valid region in the partial cases.
Within the overlap, source 0 wins ties on the pinned A2/A3 implementation;
outside it, source 0 is copied together with its paired index.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16


def _src0() -> torch.Tensor:
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(11) - 5).contiguous()


def _src1() -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(7) - 3
    values[:, ::5] = _src0()[:, ::5]
    return values.contiguous()


def _src1_below_src0() -> torch.Tensor:
    return (_src0() - 1).contiguous()


def _src1_above_src0() -> torch.Tensor:
    return (_src0() + 1).contiguous()


def _idx0() -> torch.Tensor:
    return torch.arange(M * N, dtype=torch.int32).reshape(M, N).contiguous()


def _idx1() -> torch.Tensor:
    return (1000 + torch.arange(M * N, dtype=torch.int32)).reshape(M, N).contiguous()


def _part_argmax(v_rows: int, v_cols: int):
    @pl.program
    class PartArgMax:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            idx0: pl.Tensor[[M, N], pl.INT32],
            idx1: pl.Tensor[[M, N], pl.INT32],
            value_out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            index_out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.INT32]]:
            value0 = pl.load(src0, [0, 0], [M, N], valid_shapes=[M, N])
            value1 = pl.load(src1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            index0 = pl.load(idx0, [0, 0], [M, N], valid_shapes=[M, N])
            index1 = pl.load(idx1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            value, index = pl.tile.part_argmax(value0, value1, index0, index1)
            value_out = pl.store(value, [0, 0], value_out)
            index_out = pl.store(index, [0, 0], index_out)
            return value_out, index_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            idx0: pl.Tensor[[M, N], pl.INT32],
            idx1: pl.Tensor[[M, N], pl.INT32],
            value_out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            index_out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.INT32]]:
            return self.kernel(src0, src1, idx0, idx1, value_out, index_out)

    return PartArgMax


def _part_argmin(v_rows: int, v_cols: int):
    @pl.program
    class PartArgMin:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            idx0: pl.Tensor[[M, N], pl.INT32],
            idx1: pl.Tensor[[M, N], pl.INT32],
            value_out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            index_out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.INT32]]:
            value0 = pl.load(src0, [0, 0], [M, N], valid_shapes=[M, N])
            value1 = pl.load(src1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            index0 = pl.load(idx0, [0, 0], [M, N], valid_shapes=[M, N])
            index1 = pl.load(idx1, [0, 0], [M, N], valid_shapes=[v_rows, v_cols])
            value, index = pl.tile.part_argmin(value0, value1, index0, index1)
            value_out = pl.store(value, [0, 0], value_out)
            index_out = pl.store(index, [0, 0], index_out)
            return value_out, index_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            idx0: pl.Tensor[[M, N], pl.INT32],
            idx1: pl.Tensor[[M, N], pl.INT32],
            value_out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            index_out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.INT32]]:
            return self.kernel(src0, src1, idx0, idx1, value_out, index_out)

    return PartArgMin


class PartialArgTestCase(PTOTestCase):
    __test__ = False

    def __init__(self, op_name: str, v_rows: int, v_cols: int, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._op_name = op_name
        self._v_rows = v_rows
        self._v_cols = v_cols

    def get_name(self) -> str:
        return f"{self._op_name}_{self._v_rows}x{self._v_cols}"

    def define_tensors(self) -> list[TensorSpec]:
        src1_init = _src1
        if (self._v_rows, self._v_cols) != (M, N):
            # Keep source 0 dominant in the partial case so this specifically
            # exercises preservation of its full valid region.  The aligned
            # cases exercise source selection, ties, and distinct index pairing.
            src1_init = _src1_below_src0 if self._op_name == "part_argmax" else _src1_above_src0
        return [
            TensorSpec("src0", [M, N], DataType.FP32, init_value=_src0),
            TensorSpec("src1", [M, N], DataType.FP32, init_value=src1_init),
            TensorSpec("idx0", [M, N], DataType.INT32, init_value=_idx0),
            TensorSpec("idx1", [M, N], DataType.INT32, init_value=_idx1),
            TensorSpec("value_out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("index_out", [M, N], DataType.INT32, is_output=True),
        ]

    def get_program(self) -> Any:
        factory = _part_argmax if self._op_name == "part_argmax" else _part_argmin
        return factory(self._v_rows, self._v_cols)

    def compute_expected(self, tensors, params=None):
        src0 = tensors["src0"]
        src1 = tensors["src1"]
        choose0 = src0 >= src1 if self._op_name == "part_argmax" else src0 <= src1
        choose0[self._v_rows :, :] = True
        choose0[:, self._v_cols :] = True
        tensors["value_out"][:] = torch.where(choose0, src0, src1)
        tensors["index_out"][:] = torch.where(choose0, tensors["idx0"], tensors["idx1"])


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
@pytest.mark.parametrize("op_name", ["part_argmax", "part_argmin"])
@pytest.mark.parametrize("valid_shape", [(M, N), (11, 13)], ids=["aligned", "partial"])
def test_partial_arg(test_runner, platform, op_name, valid_shape):
    result = test_runner.run(PartialArgTestCase(op_name, *valid_shape, platform=platform))
    assert result.passed, f"Test failed: {result.error}"
