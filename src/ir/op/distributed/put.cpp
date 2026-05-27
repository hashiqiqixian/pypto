/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file put.cpp
 * @brief Distributed cross-rank tensor write — ``pld.tensor.put``.
 *
 * Synchronously writes the local window-bound :class:`DistributedTensorType`
 * ``src`` into the ``peer`` rank's slice of the window-bound
 * :class:`DistributedTensorType` ``dst`` (HCCL TPUT). Both operands live at
 * the **tensor** (GM) level; ``ConvertTensorToTileOps`` lowers the public
 * tensor op to an explicit VEC staging tile plus the internal
 * ``pld.tile.put`` form before PTO codegen, so the staging tile never appears
 * on the DSL surface. The op is a sibling of
 * ``pld.tensor.alloc_window_buffer`` / ``pld.tensor.window``, not of the
 * tile-level ``pld.tile.remote_load`` (which produces a tile).
 *
 * IR signature::
 *
 *     pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
 *     pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape,
 *                    *, atomic: int) -> Unknown
 *
 * The ``atomic`` integer is the underlying value of :enum:`AtomicType`
 * (``include/pypto/ir/comm.h``); the deducer validates the int against the
 * enum range so codegen can cast back without a separate guard. The DSL
 * surface (``pld.tensor.put`` in
 * ``python/pypto/language/distributed/op/tensor_ops.py``) accepts the typed
 * Python enum and packs ``int(atomic)`` into the kwarg. Side-effect-only —
 * the op produces :class:`UnknownType`, mirroring ``pld.system.notify`` /
 * ``pld.system.wait``.
 *
 * Verifier (strict per kind-trait rules — ``As<DistributedTensorType>`` does
 * NOT match a plain :class:`TensorType`):
 *
 * * ``dst`` / ``src`` must have :class:`DistributedTensorType` — refuse plain
 *   :class:`TensorType` so a non-window-bound tensor cannot be fed into a
 *   cross-rank write.
 * * ``peer`` must be a :class:`ScalarType` expression (integer rank index).
 * * ``dst`` and ``src`` must share element type and rank.
 * * Full-slice calls require identical positive static shapes.
 * * Subregion calls carry ``dst_offsets``, ``src_offsets``, and an explicit
 *   positive static transfer ``shape`` used to size the staging VEC buffer.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

std::vector<ExprPtr> ValidatePutContract(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs,
                                         const std::string& op_name) {
  CHECK(args.size() == 3 || args.size() == 4 || args.size() == 6 || args.size() == 7)
      << op_name << " requires 3/4 full-slice args or 6/7 subregion args, got " << args.size();
  const auto& dst = args[0];
  const auto& peer = args[1];
  const auto& src = args[2];
  CHECK(dst) << op_name << " dst argument must not be null";
  CHECK(peer) << op_name << " peer argument must not be null";
  CHECK(src) << op_name << " src argument must not be null";

  auto dst_type = As<DistributedTensorType>(dst->GetType());
  CHECK(dst_type) << op_name << " dst must be a DistributedTensor (window-bound), got "
                  << dst->GetType()->TypeName();

  CHECK(IsA<ScalarType>(peer->GetType()))
      << op_name << " peer must be a scalar (rank index), got " << peer->GetType()->TypeName();

  auto src_type = As<DistributedTensorType>(src->GetType());
  CHECK(src_type) << op_name << " src must be a DistributedTensor (window-bound), got "
                  << src->GetType()->TypeName();

  // TPUT contract: dst and src cover the same element type and rank. Full-slice
  // calls require matching static shapes; subregion calls carry an explicit
  // static transfer shape used by the staging VEC buffer.
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << op_name << " dst and src must have the same element type, got dst " << dst->GetType()->TypeName()
      << " vs src " << src->GetType()->TypeName();

  const auto& dst_shape = dst_type->shape_;
  const auto& src_shape = src_type->shape_;
  CHECK(!dst_shape.empty()) << op_name << " requires at least one dimension on dst/src";
  CHECK(dst_shape.size() == src_shape.size())
      << op_name << " dst rank (" << dst_shape.size() << ") must match src rank (" << src_shape.size() << ")";
  std::vector<ExprPtr> transfer_shape;
  if (args.size() == 3 || args.size() == 4) {
    transfer_shape = dst_shape;
    for (size_t i = 0; i < dst_shape.size(); ++i) {
      auto d = As<ConstInt>(dst_shape[i]);
      auto s = As<ConstInt>(src_shape[i]);
      CHECK(d && s) << op_name
                    << " requires static (compile-time constant) shapes on dst and src; "
                       "dimension "
                    << i << " is dynamic";
      CHECK(d->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got " << d->value_;
      CHECK(d->value_ == s->value_) << op_name << " dst and src must have the same static shape; dimension "
                                    << i << " differs (dst=" << d->value_ << ", src=" << s->value_ << ")";
    }
  } else {
    const size_t extra_base = args.size() == 6 ? 3 : 4;
    auto dst_offsets = As<MakeTuple>(args[extra_base]);
    auto src_offsets = As<MakeTuple>(args[extra_base + 1]);
    auto shape_tuple = As<MakeTuple>(args[extra_base + 2]);
    CHECK(dst_offsets) << op_name << " dst_offsets must be a tuple";
    CHECK(src_offsets) << op_name << " src_offsets must be a tuple";
    CHECK(shape_tuple) << op_name << " shape must be a tuple";
    CHECK(dst_offsets->elements_.size() == dst_shape.size()) << op_name << " dst_offsets rank must match dst rank";
    CHECK(src_offsets->elements_.size() == src_shape.size()) << op_name << " src_offsets rank must match src rank";
    CHECK(shape_tuple->elements_.size() == dst_shape.size()) << op_name << " shape rank must match tensor rank";
    transfer_shape = shape_tuple->elements_;
    for (size_t i = 0; i < transfer_shape.size(); ++i) {
      auto dim = As<ConstInt>(transfer_shape[i]);
      CHECK(dim) << op_name << " shape dimensions must be static constants";
      CHECK(dim->value_ > 0) << op_name << " shape dimension " << i << " must be positive, got " << dim->value_;
    }
  }

  auto atomic_value = GetRequiredKwarg<int>(kwargs, "atomic", op_name);
  CHECK(atomic_value == static_cast<int>(AtomicType::kNone) ||
        atomic_value == static_cast<int>(AtomicType::kAdd))
      << op_name << " atomic must be AtomicType.None_ or AtomicType.Add (got int " << atomic_value << ")";
  return transfer_shape;
}

TypePtr DeducePutType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3 || args.size() == 6)
      << "pld.tensor.put requires 3 positional arguments (dst, peer, src) or 6 "
         "(dst, peer, src, dst_offsets, src_offsets, shape), but got "
      << args.size();
  ValidatePutContract(args, kwargs, "pld.tensor.put");
  // Side-effect-only — no SSA result for downstream consumers.
  return GetUnknownType();
}

TypePtr DeducePutTileType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4 || args.size() == 7)
      << "pld.tile.put requires 4 positional arguments (dst, peer, src, stage) or 7 "
         "(dst, peer, src, stage, dst_offsets, src_offsets, shape), but got "
      << args.size();
  auto transfer_shape = ValidatePutContract(args, kwargs, "pld.tile.put");

  CHECK(args[3]) << "pld.tile.put stage argument must not be null";
  auto stage_type = As<TileType>(args[3]->GetType());
  CHECK(stage_type) << "pld.tile.put stage must be a TileType, got " << args[3]->GetType()->TypeName();

  auto dst_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(stage_type->dtype_ == dst_type->dtype_)
      << "pld.tile.put stage dtype must match dst dtype, got stage=" << stage_type->dtype_.ToString()
      << " dst=" << dst_type->dtype_.ToString();

  int64_t transfer_elems = 1;
  for (const auto& dim : transfer_shape) {
    auto d = As<ConstInt>(dim);
    INTERNAL_CHECK(d) << "Internal error: pld.tile.put transfer shape was not static after ValidatePutContract";
    transfer_elems *= d->value_;
  }
  int64_t stage_elems = 1;
  for (const auto& dim : stage_type->shape_) {
    auto d = As<ConstInt>(dim);
    INTERNAL_CHECK(d) << "Internal error: pld.tile.put stage dim is not ConstInt";
    INTERNAL_CHECK(d->value_ > 0) << "Internal error: pld.tile.put stage dim not positive (" << d->value_
                                  << ")";
    stage_elems *= d->value_;
  }
  INTERNAL_CHECK(stage_elems == transfer_elems) << "Internal error: pld.tile.put stage holds " << stage_elems
                                                << " elements, expected " << transfer_elems
                                                << " (prod(transfer shape))";

  return GetUnknownType();
}

}  // namespace

// ============================================================================
// pld.tensor.put — synchronous cross-rank write into a peer rank's slice
// ============================================================================

REGISTER_OP("pld.tensor.put")
    .set_description(
        "Cross-rank put: synchronously write the local window-bound DistributedTensor `src` "
        "into the `peer` rank's slice of the window-bound DistributedTensor `dst`. `atomic` "
        "selects plain-store vs atomic-add combine semantics. Lowered by ConvertTensorToTileOps "
        "to a `tile.create`-allocated VEC staging tile plus a `pld.tile.put` call, so the "
        "staging tile flows through PyPTO's memory allocator (required at --pto-level=level3).")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local window-bound DistributedTensor source (same dtype/rank as dst)")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeducePutType);

// ============================================================================
// pld.tile.put — tile-level form with explicit VEC staging tile (post-conversion)
// ============================================================================

REGISTER_OP("pld.tile.put")
    .set_description(
        "Tile-level form of pld.tensor.put with an explicit VEC staging tile (4th arg). "
        "Created by ConvertTensorToTileOps; not user-facing.")
    .set_op_category("DistributedOp")
    .add_argument("dst", "Remote (peer) window-bound DistributedTensor destination")
    .add_argument("peer", "Peer rank index (ScalarType, integer)")
    .add_argument("src", "Local window-bound DistributedTensor source (same dtype/rank as dst)")
    .add_argument("stage", "VEC staging TileType (rows x cols == prod(transfer shape))")
    .set_attr<int>("atomic")
    .no_memory_spec()
    .f_deduce_type(DeducePutTileType);

}  // namespace ir
}  // namespace pypto
