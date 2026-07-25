# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank ring allreduce via ``pld.tensor.allreduce(mode="ring")``.

Same on-board semantics as ``test_l3_allreduce_ring.py`` — but the InCore body
calls the new composite intrinsic ``pld.tensor.allreduce(data, signal, mode="ring")``
rather than hand-rolling the 2(P−1)-step RS+AG ring loops. After
``LowerCompositeOps`` expands the intrinsic, the runtime exercises the
compiler-generated balanced-segment and UB-subchunk schedule end to end.

The ring algorithm uses O(1) HCCL windows per rank (vs. O(P) for mesh),
2(P−1) ring rounds, balanced segments for arbitrary lengths, and UB-bounded
subchunks with explicit valid-shape tails.

Signal shape for ring: ``[2 * (NR − 1), NR]`` — one row per ring round,
one cell per rank. Each row is a monotonic counter reused by the ready and
read-complete barriers of every subchunk.

ST coverage: **P=2** and **P=4**, each at non-divisible short lengths and a
length larger than UB. The stage-in/out paths are chunked independently.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

STAGE_CHUNK = 8192


def _expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Replicate the element-wise sum of all rank inputs on every rank."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden sum is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_ring_allreduce_program(n_ranks: int, size: int):
    """Build an N-rank ring allreduce program using the composite intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks
    sz = size
    total_rounds = 2 * (nr - 1)
    stage_rows = 8 if size == 1 else 1
    stage_cols = 1 if size == 1 else STAGE_CHUNK

    @pl.program
    class RingAllReduceIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, sz], pl.FP32],
            out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, sz], pl.FP32]:
            """One-call ring allreduce via ``pld.tensor.allreduce(mode="ring")``.

            The intrinsic lowers in ``LowerCompositeOps`` to the chunked
            reduce-scatter + allgather ring schedule — the user writes one
            call and the compiler emits all the per-round barriers, remote
            loads, and accumulation loops.
            """
            # Stage-in: copy local input into this rank's HCCL window slot.
            for col, (data_iter,) in pl.range(0, sz, stage_cols, init_values=(data,)):
                valid = pl.min(stage_cols, sz - col)
                local = pl.load(
                    inp,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shapes=[1, valid],
                )
                data_iter = pl.store(local, [0, col], data_iter)
                staged_data = pl.yield_(data_iter)

            # One call — the composite rebinds ``data`` (in-place semantics,
            # same as ``pl.store``) so subsequent reads see the reduced slice.
            data = pld.tensor.allreduce(staged_data, signal, op=pld.ReduceOp.Sum, mode="ring")

            # Stage-out — reduced result → local output.
            for col, (out_iter,) in pl.range(0, sz, stage_cols, init_values=(out,)):
                valid = pl.min(stage_cols, sz - col)
                acc = pl.load(
                    data,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shapes=[1, valid],
                )
                out_iter = pl.store(acc, [0, col], out_iter)
                staged_out = pl.yield_(out_iter)
            return staged_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, sz], pl.FP32],
            out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, sz], pl.FP32]:
            """Per-device orchestration wrapper around ``reduce_step``."""
            return self.reduce_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, sz], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, sz], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, sz], pl.FP32]:
            """Launch one chip orchestration per rank with shared window buffers.

            Ring signal shape is ``[2*(NR−1), NR]`` (rounds × ranks) — one
            row per ring round, one cell per rank. The lowering advances each
            row monotonically for the ready and read-complete barriers of
            every UB-sized subchunk.
            """
            data_buf = pld.alloc_window_buffer(sz * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(total_rounds * nr * pl.INT32.get_byte())

            for r in pl.range(nr):
                data = pld.window(data_buf, [1, sz], dtype=pl.FP32)
                signal = pld.window(signal_buf, [total_rounds, nr], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    signal,
                    device=r,
                )
            return outputs

    return RingAllReduceIntrinsicNRank


class TestL3TensorRingAllReduceIntrinsic:
    """L3 distributed runtime: ring allreduce via the composite intrinsic.

    Validates that the lowered ring composite produces the expected on-board
    reduction for arbitrary static lengths.
    """

    @pytest.mark.parametrize("size", [1, 17, 4097, 65537])
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_ring_allreduce_intrinsic(self, test_config, device_ids, n_ranks, size):
        """Run non-divisible and larger-than-UB ring allreduce at P=2/P=4."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"ring allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_ring_allreduce_program(n_ranks, size)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"ring allreduce intrinsic P={n_ranks} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
