# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for GM-to-Vec ``pto.mgather``."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

ROWS = 64
COLS = 32
INDEX_ROWS = 16
ELEM_M = 8
ELEM_N = 32
_PL_DTYPE = {DataType.INT32: pl.INT32, DataType.FP16: pl.FP16, DataType.FP32: pl.FP32}
_TORCH_DTYPE = {
    DataType.INT32: torch.int32,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}


class MgatherRowTestCase(PTOTestCase):
    """Row coalescing: each index selects one complete GM row."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        valid_rows: int = INDEX_ROWS,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._valid_rows = valid_rows

    def get_name(self) -> str:
        return f"mgather_row_{self._dtype.value}_v{self._valid_rows}"

    def define_tensors(self) -> list[TensorSpec]:
        dtype = _TORCH_DTYPE[self._dtype]
        table = torch.arange(ROWS * COLS).reshape(ROWS, COLS).remainder(97).to(dtype)
        indices = torch.tensor(
            [63, 2, 41, 7, 31, 0, 18, 55, 4, 29, 13, 47, 8, 38, 22, 60],
            dtype=torch.int32,
        ).reshape(1, INDEX_ROWS)
        return [
            TensorSpec("mem", [ROWS, COLS], self._dtype, init_value=table),
            TensorSpec("idx", [1, INDEX_ROWS], DataType.INT32, init_value=indices),
            TensorSpec(
                "out",
                [INDEX_ROWS, COLS],
                self._dtype,
                is_output=True,
                init_value=torch.zeros,
            ),
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        valid_rows = self._valid_rows

        @pl.program
        class MgatherRowProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[ROWS, COLS], dtype],
                idx: pl.Tensor[[1, INDEX_ROWS], pl.INT32],
                out: pl.InOut[pl.Tensor[[INDEX_ROWS, COLS], dtype]],
            ) -> pl.Tensor[[INDEX_ROWS, COLS], dtype]:
                idx_tile: pl.Tile[[1, INDEX_ROWS], pl.INT32] = pl.load(
                    idx, [0, 0], [1, INDEX_ROWS], valid_shapes=[1, valid_rows]
                )
                result: pl.Tile[[INDEX_ROWS, COLS], dtype] = pl.tile.mgather(mem, idx_tile, coalesce="row")
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mem: pl.Tensor[[ROWS, COLS], dtype],
                idx: pl.Tensor[[1, INDEX_ROWS], pl.INT32],
                out: pl.InOut[pl.Tensor[[INDEX_ROWS, COLS], dtype]],
            ) -> pl.Tensor[[INDEX_ROWS, COLS], dtype]:
                return self.kernel(mem, idx, out)

        return MgatherRowProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        expected = torch.zeros_like(tensors["out"])
        indices = tensors["idx"].reshape(-1).to(torch.int64)
        expected[: self._valid_rows] = tensors["mem"][indices[: self._valid_rows]]
        tensors["out"][:] = expected


class MgatherElemTestCase(PTOTestCase):
    """Element coalescing: every index selects one flat GM element."""

    __test__ = False

    def __init__(
        self,
        *,
        valid_shape: tuple[int, int] = (ELEM_M, ELEM_N),
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._valid_shape = valid_shape

    def get_name(self) -> str:
        rows, cols = self._valid_shape
        return f"mgather_elem_fp32_v{rows}x{cols}"

    def define_tensors(self) -> list[TensorSpec]:
        table = torch.arange(ELEM_M * ELEM_N, dtype=torch.float32).remainder(53)
        indices = torch.arange(ELEM_M * ELEM_N - 1, -1, -1, dtype=torch.int32).reshape(ELEM_M, ELEM_N)
        return [
            TensorSpec("mem", [ELEM_M * ELEM_N], DataType.FP32, init_value=table),
            TensorSpec("idx", [ELEM_M, ELEM_N], DataType.INT32, init_value=indices),
            TensorSpec(
                "out",
                [ELEM_M, ELEM_N],
                DataType.FP32,
                is_output=True,
                init_value=torch.zeros,
            ),
        ]

    def get_program(self) -> Any:
        valid_shape = list(self._valid_shape)

        @pl.program
        class MgatherElemProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[ELEM_M * ELEM_N], pl.FP32],
                idx: pl.Tensor[[ELEM_M, ELEM_N], pl.INT32],
                out: pl.InOut[pl.Tensor[[ELEM_M, ELEM_N], pl.FP32]],
            ) -> pl.Tensor[[ELEM_M, ELEM_N], pl.FP32]:
                idx_tile: pl.Tile[[ELEM_M, ELEM_N], pl.INT32] = pl.load(
                    idx, [0, 0], [ELEM_M, ELEM_N], valid_shapes=valid_shape
                )
                result: pl.Tile[[ELEM_M, ELEM_N], pl.FP32] = pl.tile.mgather(mem, idx_tile, coalesce="elem")
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mem: pl.Tensor[[ELEM_M * ELEM_N], pl.FP32],
                idx: pl.Tensor[[ELEM_M, ELEM_N], pl.INT32],
                out: pl.InOut[pl.Tensor[[ELEM_M, ELEM_N], pl.FP32]],
            ) -> pl.Tensor[[ELEM_M, ELEM_N], pl.FP32]:
                return self.kernel(mem, idx, out)

        return MgatherElemProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self._valid_shape
        expected = torch.zeros_like(tensors["out"])
        indices = tensors["idx"][:rows, :cols].to(torch.int64)
        expected[:rows, :cols] = tensors["mem"][indices]
        tensors["out"][:] = expected


class TestMgather:
    """MGATHER row/elem, dtype, index order, and valid-shape coverage."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype", [DataType.INT32, DataType.FP16, DataType.FP32])
    def test_row_dtypes_and_nonmonotonic_indices(self, test_runner, platform, dtype):
        result = test_runner.run(MgatherRowTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("valid_rows", [9, INDEX_ROWS], ids=("partial", "full"))
    def test_row_valid_shape(self, test_runner, platform, valid_rows):
        result = test_runner.run(MgatherRowTestCase(valid_rows=valid_rows, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_shape",
        [(ELEM_M, ELEM_N), (5, ELEM_N), (ELEM_M, 19), (5, 19)],
        ids=("full", "row-tail", "col-tail", "row-col-tail"),
    )
    def test_elem_valid_shape(self, test_runner, platform, valid_shape):
        result = test_runner.run(MgatherElemTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
