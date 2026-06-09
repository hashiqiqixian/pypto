# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.host.*`` distributed host-level collective op DSL wrappers."""

from pypto.ir.op.distributed import host_ops as _ir_host
from pypto.pypto_core.ir import Call, ReduceOp

from ..typing.distributed_tensor import DistributedTensor
from ._utils import _unwrap


def allreduce_(view: DistributedTensor, *, op: ReduceOp) -> Call:
    """In-place allreduce over a window-bound DistributedTensor."""
    return _ir_host.allreduce_(_unwrap(view), op)


__all__ = ["allreduce_"]
