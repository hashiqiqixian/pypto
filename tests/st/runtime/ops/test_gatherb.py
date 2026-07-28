# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for 32-byte block-offset ``pto.tgatherb``."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

ROWS = 4
OFFSETS_PER_ROW = 8
SOURCE_BLOCKS = ROWS * OFFSETS_PER_ROW
BLOCK_BYTES = 32
_PL_DTYPE = {DataType.INT32: pl.INT32, DataType.FP16: pl.FP16, DataType.FP32: pl.FP32}
_TORCH_DTYPE = {
    DataType.INT32: torch.int32,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}
_ELEMENT_BYTES = {DataType.INT32: 4, DataType.FP16: 2, DataType.FP32: 4}


def _block_indices(pattern: str) -> torch.Tensor:
    indices = torch.arange(ROWS * OFFSETS_PER_ROW, dtype=torch.int64)
    if pattern == "reverse":
        return SOURCE_BLOCKS - 1 - indices
    if pattern == "roll3":
        return (indices + 3) % SOURCE_BLOCKS
    raise ValueError(f"unknown pattern {pattern!r}")


class GatherbTestCase(PTOTestCase):
    """Gather one 32-byte source block per UINT32 byte offset."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        pattern: str = "reverse",
        valid_blocks: tuple[int, int] | None = None,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._pattern = pattern
        self._valid_blocks = valid_blocks or (ROWS, OFFSETS_PER_ROW)

    def get_name(self) -> str:
        rows, blocks = self._valid_blocks
        return f"gatherb_{self._dtype.value}_{self._pattern}_v{rows}x{blocks}blocks"

    def define_tensors(self) -> list[TensorSpec]:
        torch_dtype = _TORCH_DTYPE[self._dtype]
        block_elements = BLOCK_BYTES // _ELEMENT_BYTES[self._dtype]
        source_cols = SOURCE_BLOCKS * block_elements
        output_cols = OFFSETS_PER_ROW * block_elements
        source = torch.arange(source_cols).reshape(1, source_cols).to(torch_dtype)
        offsets = (_block_indices(self._pattern) * BLOCK_BYTES).to(torch.int32).reshape(ROWS, OFFSETS_PER_ROW)
        return [
            TensorSpec("src", [1, source_cols], self._dtype, init_value=source),
            TensorSpec("offset", [ROWS, OFFSETS_PER_ROW], DataType.UINT32, init_value=offsets),
            TensorSpec("out", [ROWS, output_cols], self._dtype, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        block_elements = BLOCK_BYTES // _ELEMENT_BYTES[self._dtype]
        source_cols = SOURCE_BLOCKS * block_elements
        output_cols = OFFSETS_PER_ROW * block_elements
        valid_blocks = list(self._valid_blocks)

        @pl.program
        class GatherbProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[1, source_cols], dtype],
                offset: pl.Tensor[[ROWS, OFFSETS_PER_ROW], pl.UINT32],
                out: pl.InOut[pl.Tensor[[ROWS, output_cols], dtype]],
            ) -> pl.Tensor[[ROWS, output_cols], dtype]:
                src_tile: pl.Tile[[1, source_cols], dtype] = pl.load(src, [0, 0], [1, source_cols])
                offset_tile: pl.Tile[[ROWS, OFFSETS_PER_ROW], pl.UINT32] = pl.load(
                    offset, [0, 0], [ROWS, OFFSETS_PER_ROW], valid_shapes=valid_blocks
                )
                result: pl.Tile[[ROWS, output_cols], dtype] = pl.tile.gatherb(src_tile, offset_tile)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[1, source_cols], dtype],
                offset: pl.Tensor[[ROWS, OFFSETS_PER_ROW], pl.UINT32],
                out: pl.InOut[pl.Tensor[[ROWS, output_cols], dtype]],
            ) -> pl.Tensor[[ROWS, output_cols], dtype]:
                return self.kernel(src, offset, out)

        return GatherbProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_blocks = self._valid_blocks
        block_elements = BLOCK_BYTES // _ELEMENT_BYTES[self._dtype]
        source = tensors["src"].reshape(SOURCE_BLOCKS, block_elements)
        indices = tensors["offset"].to(torch.int64) // BLOCK_BYTES
        expected = torch.zeros_like(tensors["out"])
        expected[:valid_rows, : valid_blocks * block_elements] = source[
            indices[:valid_rows, :valid_blocks]
        ].reshape(valid_rows, valid_blocks * block_elements)
        tensors["out"][:] = expected


class TestGatherb:
    """TGATHERB dtype, block permutation, and valid-shape branches."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP16, DataType.FP32])
    def test_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(GatherbTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("pattern", ["reverse", "roll3"])
    def test_byte_offset_patterns(self, test_runner, platform, pattern):
        result = test_runner.run(GatherbTestCase(pattern=pattern, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_blocks",
        [(3, OFFSETS_PER_ROW), (ROWS, 5), (3, 5)],
        ids=("row-tail", "col-tail", "row-col-tail"),
    )
    def test_valid_shape(self, test_runner, platform, valid_blocks):
        result = test_runner.run(GatherbTestCase(valid_blocks=valid_blocks, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
