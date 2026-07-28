# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for lower/upper ``pto.ttri`` generation."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 32
_PL_DTYPE = {DataType.INT32: pl.INT32, DataType.FP16: pl.FP16, DataType.FP32: pl.FP32}
_TORCH_DTYPE = {
    DataType.INT32: torch.int32,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}


class TriTestCase(PTOTestCase):
    """Generate a triangular mask into a full or partial destination region."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        upper: bool = False,
        diagonal: int = 0,
        valid_shape: tuple[int, int] | None = None,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._upper = upper
        self._diagonal = diagonal
        self._valid_shape = valid_shape or (M, N)

    def get_name(self) -> str:
        side = "upper" if self._upper else "lower"
        rows, cols = self._valid_shape
        return f"tri_{self._dtype.value}_{side}_d{self._diagonal}_v{rows}x{cols}".replace("-", "m")

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "out",
                [M, N],
                self._dtype,
                is_output=True,
                init_value=torch.zeros,
            )
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        upper = self._upper
        diagonal = self._diagonal
        valid_shape = list(self._valid_shape)

        @pl.program
        class TriProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                out: pl.InOut[pl.Tensor[[M, N], dtype]],
            ) -> pl.Tensor[[M, N], dtype]:
                result: pl.Tile[[M, N], dtype] = pl.tile.tri(
                    diagonal,
                    [M, N],
                    valid_shape=valid_shape,
                    dtype=dtype,
                    upper=upper,
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                out: pl.InOut[pl.Tensor[[M, N], dtype]],
            ) -> pl.Tensor[[M, N], dtype]:
                return self.kernel(out)

        return TriProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self._valid_shape
        base = torch.ones(rows, cols)
        valid = (
            torch.triu(base, diagonal=self._diagonal)
            if self._upper
            else torch.tril(base, diagonal=self._diagonal)
        )
        expected = torch.zeros((M, N), dtype=_TORCH_DTYPE[self._dtype])
        expected[:rows, :cols] = valid.to(_TORCH_DTYPE[self._dtype])
        tensors["out"][:] = expected


class TestTri:
    """TTRI dtype, side, diagonal, and valid-shape branches."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP16, DataType.FP32])
    def test_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(TriTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        ("upper", "diagonal"),
        [(False, 0), (True, 0), (False, 2), (True, -1)],
        ids=("lower-main", "upper-main", "lower-plus2", "upper-minus1"),
    )
    def test_side_and_diagonal(self, test_runner, platform, upper, diagonal):
        result = test_runner.run(TriTestCase(upper=upper, diagonal=diagonal, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_shape",
        [(9, N), (M, 21), (9, 21)],
        ids=("row-tail", "col-tail", "row-col-tail"),
    )
    def test_valid_shape(self, test_runner, platform, valid_shape):
        result = test_runner.run(TriTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
