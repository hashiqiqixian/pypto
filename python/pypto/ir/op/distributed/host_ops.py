# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR builders for host-level distributed collectives."""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ReduceOp, Span

from ...utils import _get_span_or_capture


def allreduce_(view: _ir_core.Expr, op: ReduceOp, *, span: Span | None = None) -> Call:
    """Build a ``pld.host.allreduce_(view, op=...)`` side-effect Call."""
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("pld.host.allreduce_", [view], {"op": int(op)}, actual_span)


__all__ = ["allreduce_"]
