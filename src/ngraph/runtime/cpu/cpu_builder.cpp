//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/runtime/cpu/cpu_builder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/kernel/abs.hpp"
#include "ngraph/runtime/cpu/kernel/acos.hpp"
#include "ngraph/runtime/cpu/kernel/add.hpp"
#include "ngraph/runtime/cpu/kernel/and.hpp"
#include "ngraph/runtime/cpu/kernel/asin.hpp"
#include "ngraph/runtime/cpu/kernel/atan.hpp"
#include "ngraph/runtime/cpu/kernel/broadcast.hpp"
#include "ngraph/runtime/cpu/kernel/ceil.hpp"
#include "ngraph/runtime/cpu/kernel/cos.hpp"
#include "ngraph/runtime/cpu/kernel/cosh.hpp"
#include "ngraph/runtime/cpu/kernel/cwise_pow.hpp"
#include "ngraph/runtime/cpu/kernel/divide.hpp"
#include "ngraph/runtime/cpu/kernel/equal.hpp"
#include "ngraph/runtime/cpu/kernel/exp.hpp"
#include "ngraph/runtime/cpu/kernel/floor.hpp"
#include "ngraph/runtime/cpu/kernel/greater.hpp"
#include "ngraph/runtime/cpu/kernel/greater_eq.hpp"
#include "ngraph/runtime/cpu/kernel/less.hpp"
#include "ngraph/runtime/cpu/kernel/less_eq.hpp"
#include "ngraph/runtime/cpu/kernel/log.hpp"
#include "ngraph/runtime/cpu/kernel/maximum.hpp"
#include "ngraph/runtime/cpu/kernel/minimum.hpp"
#include "ngraph/runtime/cpu/kernel/multiply.hpp"
#include "ngraph/runtime/cpu/kernel/negative.hpp"
#include "ngraph/runtime/cpu/kernel/not.hpp"
#include "ngraph/runtime/cpu/kernel/not_equal.hpp"
#include "ngraph/runtime/cpu/kernel/or.hpp"
#include "ngraph/runtime/cpu/kernel/relu.hpp"
#include "ngraph/runtime/cpu/kernel/result.hpp"
#include "ngraph/runtime/cpu/kernel/sign.hpp"
#include "ngraph/runtime/cpu/kernel/sin.hpp"
#include "ngraph/runtime/cpu/kernel/sinh.hpp"
#include "ngraph/runtime/cpu/kernel/sqrt.hpp"
#include "ngraph/runtime/cpu/kernel/subtract.hpp"
#include "ngraph/runtime/cpu/kernel/tan.hpp"
#include "ngraph/runtime/cpu/kernel/tanh.hpp"
#include "ngraph/runtime/cpu/kernel/xor.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/halide_op.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/compiler/compiler.hpp"
#endif

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Subtract)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::subtract);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Multiply)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::multiply);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Divide)
            {
                auto& functors = external_function->get_functors();
                const ngraph::op::Divide* divop = static_cast<const ngraph::op::Divide*>(node);
                std::function<void(void*, void*, void*, size_t, bool, int)> kernel;
                SELECT_KERNEL(kernel, args[0].get_element_type(), runtime::cpu::kernel::divide)
                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                bool pythondiv = divop->is_pythondiv();
                auto functor = [&,
                                kernel,
                                element_count,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                out0_buffer_index,
                                pythondiv](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                    kernel(ctx->buffer_data[arg0_buffer_index],
                           ctx->buffer_data[arg1_buffer_index],
                           ctx->buffer_data[out0_buffer_index],
                           element_count,
                           pythondiv,
                           ectx->arena);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Equal)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::equal);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::NotEqual)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::not_equal);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Greater)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::greater);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::GreaterEq)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::greater_eq);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Less)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::less);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::LessEq)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::less_eq);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::And)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto functor =
                    [&, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::cpu::kernel::logical_and(ctx->buffer_data[arg0_buffer_index],
                                                          ctx->buffer_data[arg1_buffer_index],
                                                          ctx->buffer_data[out0_buffer_index],
                                                          element_count,
                                                          ectx->arena);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Or)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto functor =
                    [&, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::cpu::kernel::logical_or(ctx->buffer_data[arg0_buffer_index],
                                                         ctx->buffer_data[arg1_buffer_index],
                                                         ctx->buffer_data[out0_buffer_index],
                                                         element_count,
                                                         ectx->arena);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Xor)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto functor =
                    [&, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::cpu::kernel::logical_xor(ctx->buffer_data[arg0_buffer_index],
                                                          ctx->buffer_data[arg1_buffer_index],
                                                          ctx->buffer_data[out0_buffer_index],
                                                          element_count,
                                                          ectx->arena);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Maximum)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::maximum);
            }
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Minimum)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::minimum);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Power)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::cwise_pow);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Abs)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::abs);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Acos)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::acos);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Asin)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::asin);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Atan)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::atan);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Ceiling)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::ceil);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Cos)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::cos);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Cosh)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::cosh);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Floor)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::floor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Negative)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::negative);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Sqrt)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sqrt);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Result)
            {
                if (args[0].get_element_type() == element::bf16)
                {
                    auto& functors = external_function->get_functors();
                    std::function<void(void*, void*, size_t, int)> kernel;

                    kernel = ngraph::runtime::cpu::kernel::result<bfloat16>;

                    auto element_count = out[0].get_size();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto functor = [&, kernel, element_count, arg0_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               element_count,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::result);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Exp)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::exp);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Log)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::log);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Not)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::logical_not);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Sign)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sign);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Sin)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sin);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Sinh)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sinh);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Tan)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::tan);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Tanh)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::tanh);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Constant)
            {
                (void)args;
                (void)out;
                auto& functors = external_function->get_functors();

                vector<size_t> dest_indices;
                for (auto& result : external_function->get_function()->get_results())
                {
                    if (result.get() == node)
                    {
                        dest_indices.push_back(external_function->get_buffer_index(
                            result->get_output_tensor(0).get_name()));
                    }
                }
                auto src_index =
                    external_function->get_buffer_index(node->get_output_tensor(0).get_name());
                auto size = node->get_output_tensor(0).size();
                auto functor = [&, dest_indices, src_index, size](CPURuntimeContext* ctx,
                                                                  CPUExecutionContext* /* ectx */) {
                    for (auto p : dest_indices)
                    {
                        memcpy(ctx->buffer_data[p], ctx->buffer_data[src_index], size);
                    }
                };
                functors.emplace_back(functor);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Add)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::add);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Subtract)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::subtract);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Multiply)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::multiply);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Divide)
            {
                const ngraph::op::Divide* divop = static_cast<const ngraph::op::Divide*>(node);
                std::function<void(void*, void*, void*, size_t, bool, int)> kernel;
                SELECT_KERNEL(kernel, node->get_input_element_type(0), runtime::cpu::kernel::divide)
                auto element_count = shape_size(node->get_shape());
                bool pythondiv = divop->is_pythondiv();
                auto functor = [&, kernel, element_count, pythondiv](
                    const std::vector<void*>& inputs, std::vector<void*>& outputs) {
                    kernel(inputs[0], inputs[1], outputs[0], element_count, pythondiv, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Minimum)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::minimum);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Maximum)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::maximum);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Abs)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::abs);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Negative)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::negative);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Relu)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::relu);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Sqrt)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::sqrt);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Floor)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::floor);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Ceiling)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::ceil);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Equal)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::equal);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::NotEqual)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::not_equal);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Greater)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::greater);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::GreaterEq)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::greater_eq);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Less)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::less);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::LessEq)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::less_eq);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::And)
            {
                auto element_count = shape_size(node->get_shape());

                auto functor = [&, element_count](const std::vector<void*>& inputs,
                                                  std::vector<void*>& outputs) {
                    runtime::cpu::kernel::logical_and(
                        inputs[0], inputs[1], outputs[0], element_count, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Or)
            {
                auto element_count = shape_size(node->get_shape());

                auto functor = [&, element_count](const std::vector<void*>& inputs,
                                                  std::vector<void*>& outputs) {
                    runtime::cpu::kernel::logical_or(
                        inputs[0], inputs[1], outputs[0], element_count, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Xor)
            {
                auto element_count = shape_size(node->get_shape());

                auto functor = [&, element_count](const std::vector<void*>& inputs,
                                                  std::vector<void*>& outputs) {
                    runtime::cpu::kernel::logical_xor(
                        inputs[0], inputs[1], outputs[0], element_count, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Sign)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::sign);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Not)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::logical_not);
            }

#define TI(x) type_index(typeid(x))

            BuildOpMap& GetGlobalBuildDispatcher()
            {
                static BuildOpMap build_dispatcher{
                    {TI(ngraph::op::Parameter), &runtime::cpu::Builder::nop},
                    {TI(ngraph::op::CompiledKernel),
                     &runtime::cpu::Builder::build<ngraph::op::CompiledKernel>},
                    {TI(ngraph::runtime::cpu::op::HalideOp),
                     &runtime::cpu::Builder::build<ngraph::runtime::cpu::op::HalideOp>}};

                return build_dispatcher;
            }

            BuildNodeExecutorMap& GetGlobalCFDispatcherCPU()
            {
                static BuildNodeExecutorMap build_cf_dispatcher_cpu{};
                return build_cf_dispatcher_cpu;
            }

            void register_cpu_builders()
            {
                REGISTER_OP_BUILDER(Constant);
                REGISTER_OP_BUILDER(Result);
                REGISTER_OP_BUILDER(Subtract);
                REGISTER_OP_BUILDER(Multiply);
                REGISTER_OP_BUILDER(Divide);
                REGISTER_OP_BUILDER(Power);
                REGISTER_OP_BUILDER(Abs);
                REGISTER_OP_BUILDER(Acos);
                REGISTER_OP_BUILDER(Asin);
                REGISTER_OP_BUILDER(Atan);
                REGISTER_OP_BUILDER(Ceiling);
                REGISTER_OP_BUILDER(Cos);
                REGISTER_OP_BUILDER(Cosh);
                REGISTER_OP_BUILDER(Floor);
                REGISTER_OP_BUILDER(Negative);
                REGISTER_OP_BUILDER(Exp);
                REGISTER_OP_BUILDER(Log);
                REGISTER_OP_BUILDER(Sqrt);
                REGISTER_OP_BUILDER(Sign);
                REGISTER_OP_BUILDER(Sin);
                REGISTER_OP_BUILDER(Sinh);
                REGISTER_OP_BUILDER(Tan);
                REGISTER_OP_BUILDER(Tanh);

                REGISTER_OP_BUILDER(Not);
                REGISTER_OP_BUILDER(Equal);
                REGISTER_OP_BUILDER(NotEqual);
                REGISTER_OP_BUILDER(Greater);
                REGISTER_OP_BUILDER(GreaterEq);
                REGISTER_OP_BUILDER(Less);
                REGISTER_OP_BUILDER(LessEq);
                REGISTER_OP_BUILDER(Maximum);
                REGISTER_OP_BUILDER(Minimum);
                REGISTER_OP_BUILDER(And);
                REGISTER_OP_BUILDER(Or);
                REGISTER_OP_BUILDER(Xor);

                REGISTER_CF_BUILDER(Add);
                REGISTER_CF_BUILDER(Subtract);
                REGISTER_CF_BUILDER(Multiply);
                REGISTER_CF_BUILDER(Divide);
                REGISTER_CF_BUILDER(Minimum);
                REGISTER_CF_BUILDER(Maximum);
                REGISTER_CF_BUILDER(Abs);
                REGISTER_CF_BUILDER(Negative);
                REGISTER_CF_BUILDER(Relu);
                REGISTER_CF_BUILDER(Sqrt);
                REGISTER_CF_BUILDER(Floor);
                REGISTER_CF_BUILDER(Ceiling);
                REGISTER_CF_BUILDER(Equal);
                REGISTER_CF_BUILDER(NotEqual);
                REGISTER_CF_BUILDER(Greater);
                REGISTER_CF_BUILDER(GreaterEq);
                REGISTER_CF_BUILDER(Less);
                REGISTER_CF_BUILDER(LessEq);
                REGISTER_CF_BUILDER(And);
                REGISTER_CF_BUILDER(Or);
                REGISTER_CF_BUILDER(Xor);
                REGISTER_CF_BUILDER(Sign);
                REGISTER_CF_BUILDER(Not);
            }
        }
    }
}
