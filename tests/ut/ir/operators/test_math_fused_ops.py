# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type contracts for TAXPY, TADDRELU, TPOW, and TPOWS."""

import pytest
from pypto import ir
from pypto.ir.op import tile_ops as tile
from pypto.pypto_core import DataType


def _tile(name, dtype=DataType.FP32, valid_shape=(7, 13)):
    view = ir.TileView(
        valid_shape=list(valid_shape),
        blayout=ir.TileLayout.row_major,
        slayout=ir.TileLayout.none_box,
    )
    return ir.Var(name, ir.TileType([8, 16], dtype, tile_view=view), ir.Span.unknown())


def test_axpy_and_add_relu_preserve_destination_contract():
    src = _tile("src", DataType.FP16)
    dst = _tile("dst", DataType.FP32)

    axpy = tile.axpy(src, 2.0, dst)
    fused = tile.add_relu(dst, _tile("rhs"))

    assert axpy.type.dtype == DataType.FP32
    assert [dim.value for dim in axpy.type.get_effective_tile_view().valid_shape] == [7, 13]
    assert fused.type.dtype == DataType.FP32


@pytest.mark.parametrize("high_precision", [False, True])
def test_float_pow_forms_require_and_accept_tmp(high_precision):
    base = _tile("base")
    exp = _tile("exp")
    tmp = _tile("tmp")

    assert tile.pow(base, exp, tmp, high_precision=high_precision).type.dtype == DataType.FP32
    assert tile.pows(base, 2.0, tmp, high_precision=high_precision).type.dtype == DataType.FP32


def test_integer_pow_forms_omit_tmp():
    base = _tile("base", DataType.INT32)
    exp = _tile("exp", DataType.INT32)

    assert tile.pow(base, exp).type.dtype == DataType.INT32
    assert tile.pows(base, 3).type.dtype == DataType.INT32


def test_pow_rejects_wrong_tmp_contract():
    base = _tile("base")
    exp = _tile("exp")

    with pytest.raises(ValueError, match="requires tmp"):
        tile.pow(base, exp)
    with pytest.raises(ValueError, match="forbids tmp"):
        tile.pow(_tile("ibase", DataType.INT32), _tile("iexp", DataType.INT32), _tile("tmp", DataType.INT32))
    with pytest.raises(ValueError, match="high_precision"):
        tile.pows(_tile("ibase", DataType.INT32), 2, high_precision=True)
