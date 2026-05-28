# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 EP dispatch/combine end-to-end test.

This mirrors ``runtime/examples/workers/l3/ep_dispatch_combine`` at the PyPTO
DSL level:

* dispatch builds per-destination expert counts from ``indices``;
* every rank publishes the global ``pub_counts`` table;
* dispatch pushes x / weight / idx payloads into ``recv_*[loc_e, slot]``;
* dispatch stages window-backed payloads into host-backed child outputs;
* local expert computes ``recv_y = cast(x * weight, BF16)`` for received rows;
* combine uses ``pub_counts`` and ``recv_idx`` to push rows back to the owner;
* the owner reduces TOPK rows into FP32 ``routed_y``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

N_RANKS = 2
T = 8
TOPK = 2
D = 64
L = 4
R = 32
W_PAD = 8
IDX_PAD = 8
N_ROUTES = T * TOPK
E_GLOBAL = N_RANKS * L
RECV_ROWS = L * R


def _build_ep_dispatch_combine_program():
    """Build the 2-rank EP dispatch/combine program at call time."""

    @pl.program
    class EPDispatchCombine:
        @pl.function(type=pl.FunctionType.InCore)
        def dispatch_step(
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_out: pl.Out[pl.Tensor[[RECV_ROWS, D], pl.BF16]],
            recv_w_out: pl.Out[pl.Tensor[[RECV_ROWS, 1], pl.FP32]],
            recv_idx_out: pl.Out[pl.Tensor[[RECV_ROWS, 1], pl.INT32]],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            send_x: pl.InOut[pld.DistributedTensor[[T, D], pl.BF16]],
            send_w: pl.InOut[pld.DistributedTensor[[N_ROUTES, W_PAD], pl.FP32]],
            send_idx: pl.InOut[pld.DistributedTensor[[N_ROUTES, IDX_PAD], pl.INT32]],
            pub_counts: pl.InOut[pld.DistributedTensor[[N_RANKS, N_RANKS, L], pl.INT32]],
            count_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            recv_x: pl.InOut[pld.DistributedTensor[[RECV_ROWS, D], pl.BF16]],
            recv_w: pl.InOut[pld.DistributedTensor[[RECV_ROWS, W_PAD], pl.FP32]],
            recv_idx: pl.InOut[pld.DistributedTensor[[RECV_ROWS, IDX_PAD], pl.INT32]],
            data_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            rank: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[RECV_ROWS, 1], pl.INT32]:
            # Stage input tensors into rank-local window-backed tensors because
            # pld.tensor.put writes from DistributedTensor slices.
            all_x = pl.load(x_norm, [0, 0], [T, D])
            all_w = pl.load(w_padded, [0, 0], [N_ROUTES, W_PAD])
            all_idx = pl.load(idx_padded, [0, 0], [N_ROUTES, IDX_PAD])
            pl.store(all_x, [0, 0], send_x)
            pl.store(all_w, [0, 0], send_w)
            pl.store(all_idx, [0, 0], send_idx)

            # histogram: send_counts[dst][loc_e], backed by local arrays.
            counts_dst0 = pl.array.create(L, pl.INT32)
            counts_dst1 = pl.array.create(L, pl.INT32)
            for e in pl.range(L):
                counts_dst0[e] = 0
                counts_dst1[e] = 0

            route_dst = pl.array.create(N_ROUTES, pl.INT32)
            route_loc_e = pl.array.create(N_ROUTES, pl.INT32)
            for r in pl.range(N_ROUTES):
                t = r // TOPK
                k = r % TOPK
                eid: pl.Scalar[pl.INT32] = pl.read(indices, [t, k])
                dst: pl.Scalar[pl.INT32] = eid // L
                loc_e: pl.Scalar[pl.INT32] = eid - dst * L
                route_dst[r] = dst
                route_loc_e[r] = loc_e
                if dst == 0:
                    counts_dst0[loc_e] = counts_dst0[loc_e] + 1
                else:
                    counts_dst1[loc_e] = counts_dst1[loc_e] + 1

            # publish: every rank gets the full pub_counts[src][dst][expert].
            for peer in pl.range(N_RANKS):
                for e in pl.range(L):
                    c0: pl.Scalar[pl.INT32] = counts_dst0[e]
                    c1: pl.Scalar[pl.INT32] = counts_dst1[e]
                    if c0 != 0:
                        pld.system.notify(
                            pub_counts,
                            peer=peer,
                            offsets=[rank, 0, e],
                            value=c0,
                            op=pld.NotifyOp.AtomicAdd,
                        )
                    if c1 != 0:
                        pld.system.notify(
                            pub_counts,
                            peer=peer,
                            offsets=[rank, 1, e],
                            value=c1,
                            op=pld.NotifyOp.AtomicAdd,
                        )
            pl.system.bar_all()
            pld.system.notify(
                count_done,
                peer=(rank + 1) % N_RANKS,
                offsets=[rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(count_done, offsets=[(rank + 1) % N_RANKS, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.bar_all()

            # prefix_sum pieces used by payload_push. Runtime sorts the route
            # table by (dst, loc_e); iterating all routes with a per-(dst,e)
            # cursor is equivalent for the visible slot order.
            cursor_dst0 = pl.array.create(L, pl.INT32)
            cursor_dst1 = pl.array.create(L, pl.INT32)
            base_dst0 = pl.array.create(L, pl.INT32)
            base_dst1 = pl.array.create(L, pl.INT32)
            for e in pl.range(L):
                cursor_dst0[e] = 0
                cursor_dst1[e] = 0
                if rank == 0:
                    base_dst0[e] = 0
                    base_dst1[e] = 0
                else:
                    base_dst0[e] = pl.read(pub_counts, [0, 0, e])
                    base_dst1[e] = pl.read(pub_counts, [0, 1, e])

            # payload_push: x / weight / idx channels.
            for r in pl.range(N_ROUTES):
                t = r // TOPK
                dst_i32: pl.Scalar[pl.INT32] = route_dst[r]
                loc_e_i32: pl.Scalar[pl.INT32] = route_loc_e[r]
                dst_idx: pl.Scalar[pl.INDEX] = pl.cast(dst_i32, pl.INDEX)
                loc_e_idx: pl.Scalar[pl.INDEX] = pl.cast(loc_e_i32, pl.INDEX)
                if dst_idx == 0:
                    slot_i32: pl.Scalar[pl.INT32] = base_dst0[loc_e_idx] + cursor_dst0[loc_e_idx]
                    cursor_dst0[loc_e_idx] = cursor_dst0[loc_e_idx] + 1
                else:
                    slot_i32: pl.Scalar[pl.INT32] = base_dst1[loc_e_idx] + cursor_dst1[loc_e_idx]
                    cursor_dst1[loc_e_idx] = cursor_dst1[loc_e_idx] + 1

                slot: pl.Scalar[pl.INDEX] = pl.cast(slot_i32, pl.INDEX)
                row = loc_e_idx * R + slot
                pld.tensor.put(
                    recv_x,
                    peer=dst_idx,
                    src=send_x,
                    dst_offsets=[row, 0],
                    src_offsets=[t, 0],
                    shape=[1, D],
                    atomic=pld.AtomicType.None_,
                )
                pld.tensor.put(
                    recv_w,
                    peer=dst_idx,
                    src=send_w,
                    dst_offsets=[row, 0],
                    src_offsets=[r, 0],
                    shape=[1, W_PAD],
                    atomic=pld.AtomicType.None_,
                )
                pld.tensor.put(
                    recv_idx,
                    peer=dst_idx,
                    src=send_idx,
                    dst_offsets=[row, 0],
                    src_offsets=[r, 0],
                    shape=[1, IDX_PAD],
                    atomic=pld.AtomicType.None_,
                )
            pl.system.bar_all()
            pld.system.notify(
                data_done,
                peer=(rank + 1) % N_RANKS,
                offsets=[rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(data_done, offsets=[(rank + 1) % N_RANKS, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.bar_all()

            # stage_out: mirror runtime's window -> host-backed child outputs.
            for e in pl.range(L):
                c0: pl.Scalar[pl.INT32] = pl.read(pub_counts, [0, rank, e])
                c1: pl.Scalar[pl.INT32] = pl.read(pub_counts, [1, rank, e])
                count: pl.Scalar[pl.INT32] = c0 + c1
                pl.write(recv_count_out, [e, 0], count)

                for slot in pl.range(R):
                    row = e * R + slot
                    x_row = pl.load(recv_x, [row, 0], [1, D])
                    pl.store(x_row, [row, 0], recv_x_out)

                    w: pl.Scalar[pl.FP32] = pl.read(recv_w, [row, 0])
                    idx: pl.Scalar[pl.INT32] = pl.read(recv_idx, [row, 0])
                    pl.write(recv_w_out, [row, 0], w)
                    pl.write(recv_idx_out, [row, 0], idx)
            pl.system.bar_all()
            return recv_idx_out

        @pl.function(type=pl.FunctionType.InCore)
        def local_expert_step(
            self,
            recv_x_out: pl.Tensor[[RECV_ROWS, D], pl.BF16],
            recv_w_out: pl.Tensor[[RECV_ROWS, 1], pl.FP32],
            recv_count_out: pl.Tensor[[L, 1], pl.INT32],
            recv_y: pl.Out[pl.Tensor[[RECV_ROWS, D], pl.BF16]],
        ) -> pl.Tensor[[RECV_ROWS, D], pl.BF16]:

            # local_expert: pure local, host-backed I/O only.
            for e in pl.range(L):
                n_rows_i32: pl.Scalar[pl.INT32] = pl.read(recv_count_out, [e, 0])
                n_rows: pl.Scalar[pl.INDEX] = pl.cast(n_rows_i32, pl.INDEX)
                for slot in pl.range(n_rows):
                    row = e * R + slot
                    w: pl.Scalar[pl.FP32] = pl.read(recv_w_out, [row, 0])
                    x_tile: pl.Tile[[1, D], pl.BF16] = pl.load(recv_x_out, [row, 0], [1, D])
                    x_tile_f32: pl.Tile[[1, D], pl.FP32] = pl.cast(x_tile, pl.FP32)
                    y_tile_f32: pl.Tile[[1, D], pl.FP32] = pl.tile.muls(x_tile_f32, w)
                    y_tile: pl.Tile[[1, D], pl.BF16] = pl.cast(y_tile_f32, pl.BF16)
                    pl.store(y_tile, [row, 0], recv_y)
            pl.system.bar_all()
            return recv_y

        @pl.function(type=pl.FunctionType.InCore)
        def combine_step(
            self,
            routed_y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            pub_counts: pl.InOut[pld.DistributedTensor[[N_RANKS, N_RANKS, L], pl.INT32]],
            recv_idx_out: pl.Tensor[[RECV_ROWS, 1], pl.INT32],
            recv_y: pl.Tensor[[RECV_ROWS, D], pl.BF16],
            send_y: pl.InOut[pld.DistributedTensor[[RECV_ROWS, D], pl.BF16]],
            routed_y_buf: pl.InOut[pld.DistributedTensor[[N_ROUTES, D], pl.BF16]],
            combine_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            rank: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            all_y = pl.load(recv_y, [0, 0], [RECV_ROWS, D])
            pl.store(all_y, [0, 0], send_y)

            # combine push: use pub_counts slabs and recv_idx to address the
            # original routed_y_buf[t * TOPK + k, :].
            for peer_rank in pl.range(N_RANKS):
                for e in pl.range(L):
                    n_i32: pl.Scalar[pl.INT32] = pl.read(pub_counts, [peer_rank, rank, e])
                    n: pl.Scalar[pl.INDEX] = pl.cast(n_i32, pl.INDEX)
                    if peer_rank == 0:
                        src_off: pl.Scalar[pl.INDEX] = 0
                    else:
                        src_off_i32: pl.Scalar[pl.INT32] = pl.read(pub_counts, [0, rank, e])
                        src_off: pl.Scalar[pl.INDEX] = pl.cast(src_off_i32, pl.INDEX)

                    for row_i in pl.range(n):
                        slot = src_off + row_i
                        row = e * R + slot
                        route: pl.Scalar[pl.INT32] = pl.read(recv_idx_out, [row, 0])
                        route_idx: pl.Scalar[pl.INDEX] = pl.cast(route, pl.INDEX)
                        pld.tensor.put(
                            routed_y_buf,
                            peer=peer_rank,
                            src=send_y,
                            dst_offsets=[route_idx, 0],
                            src_offsets=[row, 0],
                            shape=[1, D],
                            atomic=pld.AtomicType.None_,
                        )
            pl.system.bar_all()
            pld.system.notify(
                combine_done,
                peer=(rank + 1) % N_RANKS,
                offsets=[rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(combine_done, offsets=[(rank + 1) % N_RANKS, 0], expected=1, cmp=pld.WaitCmp.Ge)
            pl.system.bar_all()

            # reduce: sum TOPK BF16 rows into FP32 routed_y.
            for t in pl.range(T):
                r0 = t * TOPK
                r1 = r0 + 1
                y0_bf: pl.Tile[[1, D], pl.BF16] = pl.load(routed_y_buf, [r0, 0], [1, D])
                y1_bf: pl.Tile[[1, D], pl.BF16] = pl.load(routed_y_buf, [r1, 0], [1, D])
                y0: pl.Tile[[1, D], pl.FP32] = pl.cast(y0_bf, pl.FP32)
                y1: pl.Tile[[1, D], pl.FP32] = pl.cast(y1_bf, pl.FP32)
                summed = pl.add(y0, y1)
                pl.store(summed, [t, 0], routed_y)
            return routed_y

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_dispatch(
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_out: pl.Out[pl.Tensor[[RECV_ROWS, D], pl.BF16]],
            recv_w_out: pl.Out[pl.Tensor[[RECV_ROWS, 1], pl.FP32]],
            recv_idx_out: pl.Out[pl.Tensor[[RECV_ROWS, 1], pl.INT32]],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            send_x: pl.InOut[pld.DistributedTensor[[T, D], pl.BF16]],
            send_w: pl.InOut[pld.DistributedTensor[[N_ROUTES, W_PAD], pl.FP32]],
            send_idx: pl.InOut[pld.DistributedTensor[[N_ROUTES, IDX_PAD], pl.INT32]],
            pub_counts: pl.InOut[pld.DistributedTensor[[N_RANKS, N_RANKS, L], pl.INT32]],
            count_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            recv_x: pl.InOut[pld.DistributedTensor[[RECV_ROWS, D], pl.BF16]],
            recv_w: pl.InOut[pld.DistributedTensor[[RECV_ROWS, W_PAD], pl.FP32]],
            recv_idx: pl.InOut[pld.DistributedTensor[[RECV_ROWS, IDX_PAD], pl.INT32]],
            data_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            rank: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[RECV_ROWS, 1], pl.INT32]:
            return self.dispatch_step(
                indices,
                x_norm,
                w_padded,
                idx_padded,
                recv_x_out,
                recv_w_out,
                recv_idx_out,
                recv_count_out,
                send_x,
                send_w,
                send_idx,
                pub_counts,
                count_done,
                recv_x,
                recv_w,
                recv_idx,
                data_done,
                rank,
            )

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_local(
            self,
            recv_x_out: pl.Tensor[[RECV_ROWS, D], pl.BF16],
            recv_w_out: pl.Tensor[[RECV_ROWS, 1], pl.FP32],
            recv_count_out: pl.Tensor[[L, 1], pl.INT32],
            recv_y: pl.Out[pl.Tensor[[RECV_ROWS, D], pl.BF16]],
        ) -> pl.Tensor[[RECV_ROWS, D], pl.BF16]:
            return self.local_expert_step(recv_x_out, recv_w_out, recv_count_out, recv_y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_combine(
            self,
            routed_y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            pub_counts: pl.InOut[pld.DistributedTensor[[N_RANKS, N_RANKS, L], pl.INT32]],
            recv_idx_out: pl.Tensor[[RECV_ROWS, 1], pl.INT32],
            recv_y: pl.Tensor[[RECV_ROWS, D], pl.BF16],
            send_y: pl.InOut[pld.DistributedTensor[[RECV_ROWS, D], pl.BF16]],
            routed_y_buf: pl.InOut[pld.DistributedTensor[[N_ROUTES, D], pl.BF16]],
            combine_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            rank: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            return self.combine_step(routed_y, pub_counts, recv_idx_out, recv_y, send_y, routed_y_buf, combine_done, rank)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_ep(
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_out: pl.Out[pl.Tensor[[RECV_ROWS, D], pl.BF16]],
            recv_w_out: pl.Out[pl.Tensor[[RECV_ROWS, 1], pl.FP32]],
            recv_idx_out: pl.Out[pl.Tensor[[RECV_ROWS, 1], pl.INT32]],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            recv_y: pl.Out[pl.Tensor[[RECV_ROWS, D], pl.BF16]],
            routed_y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            send_x: pl.InOut[pld.DistributedTensor[[T, D], pl.BF16]],
            send_w: pl.InOut[pld.DistributedTensor[[N_ROUTES, W_PAD], pl.FP32]],
            send_idx: pl.InOut[pld.DistributedTensor[[N_ROUTES, IDX_PAD], pl.INT32]],
            pub_counts: pl.InOut[pld.DistributedTensor[[N_RANKS, N_RANKS, L], pl.INT32]],
            count_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            recv_x: pl.InOut[pld.DistributedTensor[[RECV_ROWS, D], pl.BF16]],
            recv_w: pl.InOut[pld.DistributedTensor[[RECV_ROWS, W_PAD], pl.FP32]],
            recv_idx: pl.InOut[pld.DistributedTensor[[RECV_ROWS, IDX_PAD], pl.INT32]],
            data_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            send_y: pl.InOut[pld.DistributedTensor[[RECV_ROWS, D], pl.BF16]],
            routed_y_buf: pl.InOut[pld.DistributedTensor[[N_ROUTES, D], pl.BF16]],
            combine_done: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
            rank: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            self.dispatch_step(
                indices,
                x_norm,
                w_padded,
                idx_padded,
                recv_x_out,
                recv_w_out,
                recv_idx_out,
                recv_count_out,
                send_x,
                send_w,
                send_idx,
                pub_counts,
                count_done,
                recv_x,
                recv_w,
                recv_idx,
                data_done,
                rank,
            )
            self.local_expert_step(recv_x_out, recv_w_out, recv_count_out, recv_y)
            return self.combine_step(routed_y, pub_counts, recv_idx_out, recv_y, send_y, routed_y_buf, combine_done, rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            indices: pl.Tensor[[N_RANKS, T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[N_RANKS, T, D], pl.BF16],
            w_padded: pl.Tensor[[N_RANKS, N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_RANKS, N_ROUTES, IDX_PAD], pl.INT32],
            outputs: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
        ) -> pl.Tensor[[N_RANKS, T, D], pl.FP32]:
            recv_x_out = pl.create_tensor([N_RANKS, RECV_ROWS, D], dtype=pl.BF16)
            recv_w_out = pl.create_tensor([N_RANKS, RECV_ROWS, 1], dtype=pl.FP32)
            recv_idx_out = pl.create_tensor([N_RANKS, RECV_ROWS, 1], dtype=pl.INT32)
            recv_count_out = pl.create_tensor([N_RANKS, L, 1], dtype=pl.INT32)
            recv_y = pl.create_tensor([N_RANKS, RECV_ROWS, D], dtype=pl.BF16)

            send_x_buf = pld.alloc_window_buffer(T * D * 2)
            send_w_buf = pld.alloc_window_buffer(N_ROUTES * W_PAD * 4)
            send_idx_buf = pld.alloc_window_buffer(N_ROUTES * IDX_PAD * 4)
            pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * L * 4)
            count_done_buf = pld.alloc_window_buffer(64)
            recv_x_buf = pld.alloc_window_buffer(L * R * D * 2)
            recv_w_buf = pld.alloc_window_buffer(L * R * W_PAD * 4)
            recv_idx_buf = pld.alloc_window_buffer(L * R * IDX_PAD * 4)
            data_done_buf = pld.alloc_window_buffer(64)
            send_y_buf = pld.alloc_window_buffer(L * R * D * 2)
            routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
            combine_done_buf = pld.alloc_window_buffer(64)

            for r in pl.range(pld.world_size()):
                send_x = pld.window(send_x_buf, [T, D], dtype=pl.BF16)
                send_w = pld.window(send_w_buf, [N_ROUTES, W_PAD], dtype=pl.FP32)
                send_idx = pld.window(send_idx_buf, [N_ROUTES, IDX_PAD], dtype=pl.INT32)
                pub_counts = pld.window(pub_counts_buf, [N_RANKS, N_RANKS, L], dtype=pl.INT32)
                count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                recv_x = pld.window(recv_x_buf, [RECV_ROWS, D], dtype=pl.BF16)
                recv_w = pld.window(recv_w_buf, [RECV_ROWS, W_PAD], dtype=pl.FP32)
                recv_idx = pld.window(recv_idx_buf, [RECV_ROWS, IDX_PAD], dtype=pl.INT32)
                data_done = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                send_y = pld.window(send_y_buf, [RECV_ROWS, D], dtype=pl.BF16)
                routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
                combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                self.chip_orch_ep(
                    indices[r],
                    x_norm[r],
                    w_padded[r],
                    idx_padded[r],
                    recv_x_out[r],
                    recv_w_out[r],
                    recv_idx_out[r],
                    recv_count_out[r],
                    recv_y[r],
                    outputs[r],
                    send_x,
                    send_w,
                    send_idx,
                    pub_counts,
                    count_done,
                    recv_x,
                    recv_w,
                    recv_idx,
                    data_done,
                    send_y,
                    routed_y_buf,
                    combine_done,
                    r,
                    device=r,
                )
            return outputs

    return EPDispatchCombine


def _generate_routing_indices(seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    while True:
        indices = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
        for r in range(N_RANKS):
            for t in range(T):
                indices[r, t, :] = torch.randperm(E_GLOBAL, generator=rng)[:TOPK].to(torch.int32)

        counts = torch.zeros(N_RANKS, L, dtype=torch.int32)
        for src in range(N_RANKS):
            for t in range(T):
                for k in range(TOPK):
                    eid = int(indices[src, t, k])
                    counts[eid // L, eid % L] += 1
        if int(counts.max()) <= R:
            return indices
        seed += 1
        rng.manual_seed(seed)


def _pack_weights_padded(weights: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(N_RANKS, N_ROUTES, W_PAD, dtype=torch.float32)
    for rank in range(N_RANKS):
        for t in range(T):
            for k in range(TOPK):
                out[rank, t * TOPK + k, 0] = weights[rank, t, k]
    return out


def _pack_idx_padded() -> torch.Tensor:
    out = torch.zeros(N_RANKS, N_ROUTES, IDX_PAD, dtype=torch.int32)
    for rank in range(N_RANKS):
        for route in range(N_ROUTES):
            out[rank, route, 0] = route
    return out


class TestL3EPDispatchCombine:
    """L3 distributed runtime: EP-style dispatch, local expert, and combine."""

    def test_dispatch_combine_roundtrip(self, test_config, device_ids):
        if len(device_ids) < N_RANKS:
            pytest.skip(f"ep dispatch/combine needs {N_RANKS} devices, got {device_ids}")

        program = _build_ep_dispatch_combine_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        )
        print(f"[ep_dispatch_combine] build output: {compiled.output_dir}", flush=True)

        x_norm = torch.tensor(
            [[[rank * 100 + t * 10 + d for d in range(D)] for t in range(T)] for rank in range(N_RANKS)],
            dtype=torch.bfloat16,
        )
        weights = torch.tensor(
            [[[(rank + 1) * 0.01 + t * 0.1 + k * 0.001 for k in range(TOPK)] for t in range(T)] for rank in range(N_RANKS)],
            dtype=torch.float32,
        )
        indices = _generate_routing_indices(seed=20260510)
        w_padded = _pack_weights_padded(weights)
        idx_padded = _pack_idx_padded()
        outputs = torch.zeros((N_RANKS, T, D), dtype=torch.float32)

        compiled(indices, x_norm, w_padded, idx_padded, outputs)

        expected = torch.zeros((N_RANKS, T, D), dtype=torch.float32)
        for rank in range(N_RANKS):
            for t in range(T):
                for k in range(TOPK):
                    weighted = weights[rank, t, k] * x_norm[rank, t].to(torch.float32)
                    expected[rank, t] += weighted.to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(outputs, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
