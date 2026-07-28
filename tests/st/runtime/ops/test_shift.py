# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact-op system tests for the tile shift families."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 64
VALID_SHAPE = (11, 47)

_PL_DT = {
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
}

_WIDTH = {
    DataType.INT16: 16,
    DataType.UINT16: 16,
}


def _values(dtype: DataType) -> torch.Tensor:
    width = _WIDTH[dtype]
    if dtype == DataType.INT16:
        values = [-32768, -17, -1, 0, 1, 17, 16383]
    else:
        values = [0, 1, 17, (1 << (width - 1)), (1 << width) - 1]
    index = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(len(values))
    result = torch.zeros((M, N), dtype=torch.int64)
    for i, value in enumerate(values):
        result[index == i] = value
    return result.to(dtype.torch_dtype).contiguous()


def _shift_counts(dtype: DataType) -> torch.Tensor:
    width = _WIDTH[dtype]
    values = [0, 1, width - 1]
    index = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(len(values))
    result = torch.zeros((M, N), dtype=torch.int64)
    for i, value in enumerate(values):
        result[index == i] = value
    return result.to(dtype.torch_dtype).contiguous()


def _make_program(op_name: str, dtype: DataType, scalar: int | None):
    pl_dtype = _PL_DT[dtype]
    valid = list(VALID_SHAPE)

    if op_name == "shl":

        @pl.program
        class ShlProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
                shift_tile = pl.load(shift, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.shl(src_tile, shift_tile), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src, shift, out)

        return ShlProgram

    if op_name == "shr":

        @pl.program
        class ShrProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
                shift_tile = pl.load(shift, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.shr(src_tile, shift_tile), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src, shift, out)

        return ShrProgram

    assert scalar is not None

    if op_name == "shls":

        @pl.program
        class ShlsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.shls(src_tile, scalar), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src, out)

        return ShlsProgram

    assert op_name == "shrs"

    @pl.program
    class ShrsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
            return pl.store(pl.tile.shrs(src_tile, scalar), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            return self.kernel(src, out)

    return ShrsProgram


class ShiftCase(PTOTestCase):
    """One direct shift-instruction case."""

    __test__ = False

    def __init__(self, *, op_name: str, dtype: DataType, scalar: int | None = None):
        super().__init__()
        self.op_name = op_name
        self.dtype = dtype
        self.scalar = scalar

    def get_name(self) -> str:
        scalar_tag = f"_s{self.scalar}" if self.scalar is not None else ""
        return f"tile_{self.op_name}_{self.dtype.value}_v{VALID_SHAPE[0]}x{VALID_SHAPE[1]}{scalar_tag}"

    def define_tensors(self) -> list[TensorSpec]:
        specs = [TensorSpec("src", [M, N], self.dtype, init_value=lambda: _values(self.dtype))]
        if self.scalar is None:
            specs.append(
                TensorSpec("shift", [M, N], self.dtype, init_value=lambda: _shift_counts(self.dtype))
            )
        specs.append(TensorSpec("out", [M, N], self.dtype, init_value=torch.zeros, is_output=True))
        return specs

    def get_program(self) -> Any:
        return _make_program(self.op_name, self.dtype, self.scalar)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = VALID_SHAPE
        width = _WIDTH[self.dtype]
        src = tensors["src"][:rows, :cols].to(torch.int64)
        shift: torch.Tensor | int
        if self.scalar is None:
            shift = tensors["shift"][:rows, :cols].to(torch.int64)
        else:
            shift = self.scalar

        if self.op_name in {"shl", "shls"}:
            mask = (1 << width) - 1
            value = torch.bitwise_left_shift(src, shift) & mask
            if self.dtype == DataType.INT16:
                sign = 1 << (width - 1)
                value = (value ^ sign) - sign
        else:
            if self.dtype == DataType.UINT16:
                src = src & ((1 << width) - 1)
            value = torch.bitwise_right_shift(src, shift)

        expected = torch.zeros_like(tensors["out"])
        expected[:rows, :cols] = value.to(self.dtype.torch_dtype)
        tensors["out"][:] = expected


_CASES = [
    *[
        pytest.param(op_name, dtype, None, id=f"t{op_name}-{dtype.value}-counts-0-1-15")
        for op_name in ("shl", "shr")
        for dtype in (DataType.INT16, DataType.UINT16)
    ],
    *[
        pytest.param(op_name, DataType.INT16, scalar, id=f"t{op_name}-int16-s{scalar}")
        for op_name in ("shls", "shrs")
        for scalar in (0, 15)
    ],
]


class TestShiftFamily:
    """A2/A3 same-name coverage for four tile shift instructions."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("op_name,dtype,scalar", _CASES)
    def test_shift(self, test_runner, op_name, dtype, scalar):
        result = test_runner.run(ShiftCase(op_name=op_name, dtype=dtype, scalar=scalar))
        assert result.passed, f"Test failed: {result.error}"
