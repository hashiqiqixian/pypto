# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime coverage for TAXPY, TADDRELU, TPOW, and TPOWS."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16
VALID = [11, 13]


def _src0() -> torch.Tensor:
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(9) / 4 + 0.5).contiguous()


def _src1() -> torch.Tensor:
    return (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(13) / 3 - 2).contiguous()


def _axpy_program():
    @pl.program
    class AxpyProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            src = pl.load(src0, [0, 0], [M, N], valid_shapes=VALID)
            dst = pl.load(src1, [0, 0], [M, N], valid_shapes=VALID)
            result = pl.tile.axpy(src, 2.0, dst)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            return self.kernel(src0, src1, out)

    return AxpyProgram


def _add_relu_program():
    @pl.program
    class AddReluProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            lhs = pl.load(src0, [0, 0], [M, N], valid_shapes=VALID)
            rhs = pl.load(src1, [0, 0], [M, N], valid_shapes=VALID)
            result = pl.tile.add_relu(lhs, rhs)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            return self.kernel(src0, src1, out)

    return AddReluProgram


def _pow_program():
    @pl.program
    class PowProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            base = pl.load(src0, [0, 0], [M, N], valid_shapes=VALID)
            exp = pl.load(src1, [0, 0], [M, N], valid_shapes=VALID)
            tmp_raw = pl.tile.create([M, N], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            tmp = pl.tile.set_validshape(tmp_raw, 11, 13)
            result = pl.tile.pow(base, exp, tmp)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            return self.kernel(src0, src1, out)

    return PowProgram


def _pows_program():
    @pl.program
    class PowsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            base = pl.load(src0, [0, 0], [M, N], valid_shapes=VALID)
            tmp_raw = pl.tile.create([M, N], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            tmp = pl.tile.set_validshape(tmp_raw, 11, 13)
            result = pl.tile.pows(base, 2.0, tmp)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl.FP32],
            src1: pl.Tensor[[M, N], pl.FP32],
            out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            return self.kernel(src0, src1, out)

    return PowsProgram


class MathFusedCase(PTOTestCase):
    __test__ = False

    def __init__(self, op_name: str, *, platform: str):
        super().__init__(platform=platform)
        self.op_name = op_name

    def get_name(self) -> str:
        return f"math_fused_{self.op_name}_partial"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src0", [M, N], DataType.FP32, init_value=_src0),
            TensorSpec("src1", [M, N], DataType.FP32, init_value=_src1),
            TensorSpec("out", [M, N], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        if self.op_name == "axpy":
            return _axpy_program()
        if self.op_name == "add_relu":
            return _add_relu_program()
        if self.op_name == "pow":
            return _pow_program()
        return _pows_program()

    def compute_expected(self, tensors, params=None):
        src0 = tensors["src0"][: VALID[0], : VALID[1]]
        src1 = tensors["src1"][: VALID[0], : VALID[1]]
        if self.op_name == "axpy":
            expected = src1 + src0 * 2
        elif self.op_name == "add_relu":
            expected = torch.relu(src0 + src1)
        elif self.op_name == "pow":
            expected = torch.pow(src0, src1)
        else:
            expected = torch.pow(src0, 2)
        tensors["out"].zero_()
        tensors["out"][: VALID[0], : VALID[1]] = expected


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
@pytest.mark.parametrize("op_name", ["axpy", "add_relu", "pow", "pows"])
def test_math_fused(test_runner, platform, op_name):
    result = test_runner.run(MathFusedCase(op_name, platform=platform))
    assert result.passed, f"Test failed: {result.error}"
