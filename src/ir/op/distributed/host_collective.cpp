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
 * @file host_collective.cpp
 * @brief Host-level collective ops.
 */

#include <any>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

int GetRequiredIntKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs,
                        const std::string& key, const std::string& op_name) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<int>(v, "kwarg key: " + key);
    }
  }
  throw ValueError("Missing kwarg '" + key + "' on " + op_name);
}

DataType GetRequiredDTypeKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  for (const auto& [key, value] : kwargs) {
    if (key != "dtype") continue;
    if (value.type() == typeid(DataType)) {
      return AnyCast<DataType>(value, "kwarg key: dtype");
    }
    if (value.type() == typeid(int)) {
      return DataType(static_cast<uint8_t>(AnyCast<int>(value, "kwarg key: dtype")));
    }
  }
  throw ValueError("Missing kwarg 'dtype' on " + op_name);
}

void CheckReduceSumOnly(int op_value, const std::string& op_name) {
  CHECK(op_value == static_cast<int>(ReduceOp::kSum))
      << op_name << " currently supports only ReduceOp.Sum (got int " << op_value << ")";
}

void CheckStaticPositiveShape(const DistributedTensorTypePtr& type, const std::string& op_name) {
  int64_t numel = 1;
  for (size_t i = 0; i < type->shape_.size(); ++i) {
    auto dim = As<ConstInt>(type->shape_[i]);
    CHECK(dim) << op_name << " currently requires a static shape; dim " << i
               << " is not a ConstInt";
    CHECK(dim->value_ > 0) << op_name << " requires positive shape dims; dim " << i
                           << " is " << dim->value_;
    numel *= dim->value_;
  }
  CHECK(numel <= 256) << op_name << " currently supports at most 256 FP32 elements, got " << numel;
}

TypePtr DeduceHostAllreduceType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "pld.host.allreduce_ requires exactly 1 positional argument "
                             "(view: DistributedTensor), but got "
                          << args.size();
  CHECK(args[0]) << "pld.host.allreduce_ argument must not be null";
  auto dist_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(dist_type) << "pld.host.allreduce_ view must be a DistributedTensor produced by pld.window, got "
                   << args[0]->GetType()->TypeName();

  CheckStaticPositiveShape(dist_type, "pld.host.allreduce_");
  CheckReduceSumOnly(GetRequiredIntKwarg(kwargs, "op", "pld.host.allreduce_"), "pld.host.allreduce_");
  return GetUnknownType();
}

TypePtr DeduceBuiltinAllreduceType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "builtin.allreduce requires exactly 2 positional arguments "
                             "(data, signal), but got "
                          << args.size();
  CHECK(args[0] && args[1]) << "builtin.allreduce arguments must not be null";

  auto data_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(data_type) << "builtin.allreduce data must be a DistributedTensor, got "
                   << args[0]->GetType()->TypeName();
  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << "builtin.allreduce signal must be a DistributedTensor, got "
                     << args[1]->GetType()->TypeName();

  CheckReduceSumOnly(GetRequiredIntKwarg(kwargs, "op", "builtin.allreduce"), "builtin.allreduce");
  auto dtype = GetRequiredDTypeKwarg(kwargs, "builtin.allreduce");
  CHECK(dtype == data_type->dtype_) << "builtin.allreduce dtype kwarg must match data dtype";
  CHECK(dtype == DataType::FP32) << "builtin.allreduce currently supports only FP32 Sum";
  CHECK(signal_type->dtype_ == DataType::INT32) << "builtin.allreduce signal must be INT32";
  return GetUnknownType();
}

}  // namespace

REGISTER_OP("pld.host.allreduce_")
    .set_description("Host-level in-place allreduce over a window-bound DistributedTensor.")
    .set_op_category("DistributedOp")
    .add_argument("view", "Window-bound DistributedTensor to reduce in place")
    .set_attr<int>("op")
    .no_memory_spec()
    .f_deduce_type(DeduceHostAllreduceType);

REGISTER_OP("builtin.allreduce")
    .set_description("Compiler-internal per-chip allreduce builtin dispatch.")
    .set_op_category("BuiltinChipDispatch")
    .add_argument("data", "Window-bound DistributedTensor payload")
    .add_argument("signal", "Implicit INT32 signal DistributedTensor")
    .set_attr<int>("op")
    .set_attr<DataType>("dtype")
    .set_internal_only()
    .set_template_dir(":pypto.runtime.builtins.collectives.allreduce")
    .no_memory_spec()
    .f_deduce_type(DeduceBuiltinAllreduceType);

}  // namespace ir
}  // namespace pypto
