# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 simulator coverage for the cumulative ``thistogram`` instruction."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 8
N = 16


def _src16() -> torch.Tensor:
    rows = torch.arange(M, dtype=torch.int32).reshape(M, 1)
    cols = torch.arange(N, dtype=torch.int32).reshape(1, N)
    return ((rows << 8) | ((cols * 17 + rows) & 0xFF)).to(torch.uint16).contiguous()


def _idx16() -> torch.Tensor:
    return torch.arange(M, dtype=torch.uint8).reshape(1, M).contiguous()


def _src32() -> torch.Tensor:
    cols = torch.arange(N, dtype=torch.int64).reshape(1, N)
    low = (cols * 13 + 5) & 0xFF
    return ((0x12 << 24) | (0x34 << 16) | (0x56 << 8) | low).to(torch.uint32).contiguous()


def _idx32(rows: int) -> torch.Tensor:
    values = torch.tensor([0x56, 0x34, 0x12], dtype=torch.uint8).reshape(3, 1)
    return values[:rows].expand(rows, N).contiguous()


def _histogram16(byte: int):
    @pl.program
    class Histogram16:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl.UINT16],
            idx: pl.Tensor[[1, M], pl.UINT8],
            out: pl.Out[pl.Tensor[[M, 256], pl.UINT32]],
        ) -> pl.Tensor[[M, 256], pl.UINT32]:
            src_tile = pl.load(src, [0, 0], [M, N])
            idx_row = pl.load(idx, [0, 0], [1, M])
            idx_col = pl.tile.reshape(idx_row, [M, 1])
            result = pl.tile.histogram(src_tile, idx_col, byte=byte)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl.UINT16],
            idx: pl.Tensor[[1, M], pl.UINT8],
            out: pl.Out[pl.Tensor[[M, 256], pl.UINT32]],
        ) -> pl.Tensor[[M, 256], pl.UINT32]:
            return self.kernel(src, idx, out)

    return Histogram16


def _histogram32(byte: int, idx_rows: int):
    @pl.program
    class Histogram32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[1, N], pl.UINT32],
            idx: pl.Tensor[[idx_rows, N], pl.UINT8],
            out: pl.Out[pl.Tensor[[1, 256], pl.UINT32]],
        ) -> pl.Tensor[[1, 256], pl.UINT32]:
            src_tile = pl.load(src, [0, 0], [1, N])
            idx_tile = pl.load(idx, [0, 0], [idx_rows, N])
            result = pl.tile.histogram(src_tile, idx_tile, byte=byte)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[1, N], pl.UINT32],
            idx: pl.Tensor[[idx_rows, N], pl.UINT8],
            out: pl.Out[pl.Tensor[[1, 256], pl.UINT32]],
        ) -> pl.Tensor[[1, 256], pl.UINT32]:
            return self.kernel(src, idx, out)

    return Histogram32


def _cumulative(values: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(values.to(torch.int64), minlength=256)
    return torch.cumsum(counts, dim=0).to(torch.uint32)


class HistogramTestCase(PTOTestCase):
    __test__ = False

    def __init__(self, dtype: DataType, byte: int, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._dtype = dtype
        self._byte = byte

    def get_name(self) -> str:
        dtype_name = "uint16" if self._dtype == DataType.UINT16 else "uint32"
        return f"histogram_{dtype_name}_byte{self._byte}"

    def define_tensors(self) -> list[TensorSpec]:
        if self._dtype == DataType.UINT16:
            return [
                TensorSpec("src", [M, N], DataType.UINT16, init_value=_src16),
                TensorSpec("idx", [1, M], DataType.UINT8, init_value=_idx16),
                TensorSpec("out", [M, 256], DataType.UINT32, is_output=True),
            ]
        rows = 3 if self._byte == 0 else 2 if self._byte == 1 else 1
        return [
            TensorSpec("src", [1, N], DataType.UINT32, init_value=_src32),
            TensorSpec("idx", [rows, N], DataType.UINT8, init_value=lambda: _idx32(rows)),
            TensorSpec("out", [1, 256], DataType.UINT32, is_output=True),
        ]

    def get_program(self) -> Any:
        if self._dtype == DataType.UINT16:
            return _histogram16(self._byte)
        rows = 3 if self._byte == 0 else 2 if self._byte == 1 else 1
        return _histogram32(self._byte, rows)

    def compute_expected(self, tensors, params=None):
        src = tensors["src"].to(torch.int64)
        if self._dtype == DataType.UINT16:
            for row in range(M):
                values = (src[row] >> (8 * self._byte)) & 0xFF
                if self._byte == 0:
                    values = values[((src[row] >> 8) & 0xFF) == row]
                tensors["out"][row] = _cumulative(values)
            return

        values = (src[0] >> (8 * self._byte)) & 0xFF
        if self._byte < 3:
            for filter_byte in range(self._byte + 1, 4):
                idx_row = 3 - filter_byte
                values = values[
                    ((src[0] >> (8 * filter_byte)) & 0xFF)
                    == tensors["idx"][idx_row, 0].to(torch.int64)
                ]
        tensors["out"][0] = _cumulative(values)


@pytest.mark.platforms("a5sim")
@pytest.mark.parametrize("platform", [pytest.param("a5sim", id="a5sim")])
@pytest.mark.parametrize("byte", [0, 1])
def test_histogram_uint16(test_runner, platform, byte):
    result = test_runner.run(HistogramTestCase(DataType.UINT16, byte, platform=platform))
    assert result.passed, f"Test failed: {result.error}"


@pytest.mark.platforms("a5sim")
@pytest.mark.parametrize("platform", [pytest.param("a5sim", id="a5sim")])
@pytest.mark.parametrize("byte", [0, 1, 2, 3])
def test_histogram_uint32(test_runner, platform, byte):
    result = test_runner.run(HistogramTestCase(DataType.UINT32, byte, platform=platform))
    assert result.passed, f"Test failed: {result.error}"
