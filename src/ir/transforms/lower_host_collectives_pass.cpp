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

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  return func && func->level_.has_value() && *func->level_ == Level::HOST && func->role_.has_value() &&
         *func->role_ == Role::Orchestrator;
}

[[nodiscard]] ExprPtr ConstI64(int64_t value, Span span) {
  return std::make_shared<const ConstInt>(value, DataType::INT64, std::move(span));
}

[[nodiscard]] ExprPtr ConstIndex(int64_t value, Span span) {
  return std::make_shared<const ConstInt>(value, DataType::INDEX, std::move(span));
}

[[nodiscard]] MakeTuplePtr ShapeTuple(std::vector<ExprPtr> dims, Span span) {
  return std::make_shared<const MakeTuple>(std::move(dims), std::move(span));
}

[[nodiscard]] size_t GroupIdxForWindowBuffer(const ProgramPtr& program, const WindowBufferPtr& wb,
                                             const Span& span) {
  for (size_t gi = 0; gi < program->comm_groups_.size(); ++gi) {
    const auto& group = program->comm_groups_[gi];
    if (!group) continue;
    for (const auto& slot : group->slots_) {
      if (slot.get() == wb.get()) return gi;
    }
  }
  throw pypto::ValueError("LowerHostCollectives: DistributedTensor window buffer is not in any CommGroup at " +
                          span.to_string());
}

class HostCollectiveLowerer : public IRMutator {
 public:
  explicit HostCollectiveLowerer(const ProgramPtr& program) : program_(program) {}

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call || !call->op_ || call->op_->name_ != "pld.host.allreduce_") {
      return IRMutator::VisitStmt_(op);
    }
    modified_ = true;
    return LowerAllreduce(call, op->span_, op->leading_comments_);
  }

  [[nodiscard]] bool modified() const { return modified_; }
  [[nodiscard]] const std::vector<std::pair<size_t, WindowBufferPtr>>& added_signal_slots() const {
    return added_signal_slots_;
  }

 private:
  StmtPtr LowerAllreduce(const CallPtr& call, const Span& span,
                         const std::vector<std::string>& leading_comments) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 1, span)
        << "LowerHostCollectives: pld.host.allreduce_ must have one argument";
    auto data_type = As<DistributedTensorType>(call->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(data_type, span)
        << "LowerHostCollectives: pld.host.allreduce_ argument must be DistributedTensorType";
    INTERNAL_CHECK_SPAN(data_type->window_buffer_.has_value(), span)
        << "LowerHostCollectives requires CollectCommGroups to populate DistributedTensorType.window_buffer_";

    const auto& data_wb = data_type->window_buffer_.value();
    size_t group_idx = GroupIdxForWindowBuffer(program_, data_wb, span);
    const auto& group = program_->comm_groups_[group_idx];

    const std::string signal_suffix = std::to_string(signal_counter_++);
    const std::string signal_buf_name = "__allreduce_signal_buf_" + signal_suffix;
    const std::string signal_name = "__allreduce_signal_" + signal_suffix;
    auto signal_base = std::make_shared<const Var>(signal_buf_name, GetPtrType(), span);
    auto world_size_call = OpRegistry::GetInstance().Create("pld.system.world_size", {}, span);
    auto signal_size =
        std::make_shared<const Mul>(world_size_call, ConstI64(4, span), DataType::INT64, span);
    auto signal_wb = std::make_shared<const WindowBuffer>(signal_base, signal_size,
                                                          /*load_from_host=*/false,
                                                          /*store_to_host=*/false, span);
    auto signal_type = std::make_shared<DistributedTensorType>(
        std::vector<ExprPtr>{ConstI64(1, span), world_size_call}, DataType::INT32, signal_wb);
    auto signal_view = std::make_shared<const Var>(signal_name, signal_type, span);

    std::vector<StmtPtr> stmts;
    stmts.reserve(3);

    auto alloc_call = OpRegistry::GetInstance().Create(
        "pld.tensor.alloc_window_buffer", {signal_size}, {{"name", signal_buf_name}},
        span);
    stmts.push_back(std::make_shared<const AssignStmt>(signal_base, alloc_call, span, leading_comments));

    auto raw_window_call = OpRegistry::GetInstance().Create(
        "pld.tensor.window", {signal_base, ShapeTuple({ConstI64(1, span), world_size_call}, span)},
        {{"dtype", DataType::INT32}}, span);
    auto window_call = std::make_shared<const Call>(raw_window_call->op_, raw_window_call->args_,
                                                    raw_window_call->kwargs_, signal_type, span);
    stmts.push_back(std::make_shared<const AssignStmt>(signal_view, window_call, span));

    auto op_value = call->GetKwarg<int>("op");
    std::vector<StmtPtr> builtin_stmts;
    if (group->devices_.empty()) {
      auto r = std::make_shared<const Var>("__allreduce_rank", std::make_shared<ScalarType>(DataType::INDEX),
                                           span);
      auto builtin = OpRegistry::GetInstance().Create(
          "builtin.allreduce", {call->args_[0], signal_view},
          {{"op", op_value}, {"dtype", data_type->dtype_}}, span);
      auto with_device = std::make_shared<const Call>(
          builtin->op_, builtin->args_, builtin->kwargs_,
          std::vector<std::pair<std::string, std::any>>{{kAttrDevice, std::static_pointer_cast<const Expr>(r)}},
          builtin->GetType(), span);
      auto body = std::make_shared<const EvalStmt>(with_device, span);
      stmts.push_back(std::make_shared<const ForStmt>(
          r, ConstIndex(0, span), world_size_call, ConstIndex(1, span), std::vector<IterArgPtr>{}, body,
          std::vector<VarPtr>{}, span));
    } else {
      for (auto device : group->devices_) {
        auto builtin = OpRegistry::GetInstance().Create(
            "builtin.allreduce", {call->args_[0], signal_view},
            {{"op", op_value}, {"dtype", data_type->dtype_}}, span);
        auto with_device = std::make_shared<const Call>(
            builtin->op_, builtin->args_, builtin->kwargs_,
            std::vector<std::pair<std::string, std::any>>{
                {kAttrDevice, std::static_pointer_cast<const Expr>(ConstI64(device, span))}},
            builtin->GetType(), span);
        builtin_stmts.push_back(std::make_shared<const EvalStmt>(with_device, span));
      }
      stmts.push_back(SeqStmts::Flatten(std::move(builtin_stmts), span));
    }

    added_signal_slots_.emplace_back(group_idx, signal_wb);

    return SeqStmts::Flatten(std::move(stmts), span);
  }

  ProgramPtr program_;
  bool modified_{false};
  int signal_counter_{0};
  std::vector<std::pair<size_t, WindowBufferPtr>> added_signal_slots_;
};

ProgramPtr LowerProgram(const ProgramPtr& program) {
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  bool modified = false;
  std::vector<CommGroupPtr> groups = program->comm_groups_;

  for (const auto& [gvar, func] : program->functions_) {
    if (!IsHostOrch(func) || !func->body_) {
      new_functions[gvar] = func;
      continue;
    }
    HostCollectiveLowerer lowerer(program);
    auto new_body = lowerer.VisitStmt(func->body_);
    if (!lowerer.modified()) {
      new_functions[gvar] = func;
      continue;
    }
    modified = true;
    auto new_func = MutableCopy(func);
    new_func->body_ = new_body;
    new_functions[gvar] = new_func;
    for (const auto& [group_idx, slot] : lowerer.added_signal_slots()) {
      INTERNAL_CHECK(group_idx < groups.size()) << "LowerHostCollectives: group index out of range";
      auto new_slots = groups[group_idx]->slots_;
      new_slots.push_back(slot);
      groups[group_idx] =
          std::make_shared<const CommGroup>(groups[group_idx]->devices_, std::move(new_slots),
                                            groups[group_idx]->span_);
    }
  }

  if (!modified) return program;
  return std::make_shared<Program>(std::move(new_functions), std::move(groups), program->name_,
                                   program->span_);
}

}  // namespace

namespace pass {

Pass LowerHostCollectives() {
  PassProperties props{.required = {IRProperty::CommGroupsCollected},
                       .produced = {IRProperty::CommGroupsCollected}};
  return CreateProgramPass(LowerProgram, "LowerHostCollectives", props);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
