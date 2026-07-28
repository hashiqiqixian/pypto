# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for byte-offset ``pto.tgatherb``."""

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
_ELEMENT_BYTES = {DataType.INT32: 4, DataType.FP16: 2, DataType.FP32: 4}


def _flat_indices(pattern: str) -> torch.Tensor:
    indices = torch.arange(M * N, dtype=torch.int64)
    if pattern == "reverse":
        return M * N - 1 - indices
    if pattern == "roll7":
        return (indices + 7) % (M * N)
    raise ValueError(f"unknown pattern {pattern!r}")


class GatherbTestCase(PTOTestCase):
    """Gather one element per UINT32 byte offset."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        pattern: str = "reverse",
        valid_shape: tuple[int, int] | None = None,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._pattern = pattern
        self._valid_shape = valid_shape or (M, N)

    def get_name(self) -> str:
        rows, cols = self._valid_shape
        return f"gatherb_{self._dtype.value}_{self._pattern}_v{rows}x{cols}"

    def define_tensors(self) -> list[TensorSpec]:
        torch_dtype = _TORCH_DTYPE[self._dtype]
        source = torch.arange(M * N).reshape(M, N).to(torch_dtype)
        offsets = (_flat_indices(self._pattern) * _ELEMENT_BYTES[self._dtype]).to(torch.int32).reshape(M, N)
        return [
            TensorSpec("src", [M, N], self._dtype, init_value=source),
            TensorSpec("offset", [M, N], DataType.UINT32, init_value=offsets),
            TensorSpec("out", [M, N], self._dtype, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        valid_shape = list(self._valid_shape)

        @pl.program
        class GatherbProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], dtype],
                offset: pl.Tensor[[M, N], pl.UINT32],
                out: pl.InOut[pl.Tensor[[M, N], dtype]],
            ) -> pl.Tensor[[M, N], dtype]:
                src_tile: pl.Tile[[M, N], dtype] = pl.load(src, [0, 0], [M, N])
                offset_tile: pl.Tile[[M, N], pl.UINT32] = pl.load(
                    offset, [0, 0], [M, N], valid_shapes=valid_shape
                )
                result: pl.Tile[[M, N], dtype] = pl.tile.gatherb(src_tile, offset_tile)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], dtype],
                offset: pl.Tensor[[M, N], pl.UINT32],
                out: pl.InOut[pl.Tensor[[M, N], dtype]],
            ) -> pl.Tensor[[M, N], dtype]:
                return self.kernel(src, offset, out)

        return GatherbProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self._valid_shape
        source = tensors["src"].reshape(-1)
        indices = tensors["offset"].to(torch.int64) // _ELEMENT_BYTES[self._dtype]
        expected = torch.zeros_like(tensors["out"])
        expected[:rows, :cols] = source[indices[:rows, :cols]]
        tensors["out"][:] = expected


class TestGatherb:
    """TGATHERB dtype, byte permutation, and valid-shape branches."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP16, DataType.FP32])
    def test_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(GatherbTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("pattern", ["reverse", "roll7"])
    def test_byte_offset_patterns(self, test_runner, platform, pattern):
        result = test_runner.run(GatherbTestCase(pattern=pattern, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_shape",
        [(9, N), (M, 21), (9, 21)],
        ids=("row-tail", "col-tail", "row-col-tail"),
    )
    def test_valid_shape(self, test_runner, platform, valid_shape):
        result = test_runner.run(GatherbTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
