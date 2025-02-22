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

#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Rnn)
            {
                if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error(
                        "Rnn is supported only through MKLDNN and doesnt have reference "
                        "INTERPRETER implementation");
                }

                auto& functors = external_function->get_functors();

                auto src_layer_buffer_index =
                    external_function->get_buffer_index(args[0].get_name());
                auto src_iter_buffer_index =
                    external_function->get_buffer_index(args[1].get_name());
                auto dst_layer_buffer_index =
                    external_function->get_buffer_index(out[0].get_name());
                auto dst_iter_buffer_index = external_function->get_buffer_index(out[1].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto rnn_desc =
                    mkldnn_emitter->get_rnn_forward_desc<ngraph::op::Rnn>(node, args, out);

#if MKLDNN_VERSION_MAJOR < 1
                auto weights_layer_buffer_index =
                    external_function->get_buffer_index(args[2].get_name());
                auto weights_iter_buffer_index =
                    external_function->get_buffer_index(args[3].get_name());
                auto bias_buffer_index = external_function->get_buffer_index(args[4].get_name());

                // Rnn needs 9 primitives: src_layer, src_iter, weights_layer, weights_iter, bias,
                // dst_layer, dst_iter, workspace, and rnn_forward.
                // It needs a new workspace.
                auto rnn_index =
                    mkldnn_emitter->reserve_primitive_space(9, true /* new workspace */);
                auto& deps = mkldnn_emitter->get_primitive_deps(rnn_index);

                auto functor = [&,
                                rnn_desc,
                                rnn_index,
                                src_layer_buffer_index,
                                src_iter_buffer_index,
                                weights_layer_buffer_index,
                                weights_iter_buffer_index,
                                bias_buffer_index,
                                dst_layer_buffer_index,
                                dst_iter_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_rnn_forward(ctx->mkldnn_memories,
                                                          ctx->mkldnn_primitives,
                                                          ctx->mkldnn_scratchpad_mds,
                                                          ctx->mkldnn_workspaces,
                                                          rnn_desc,
                                                          deps,
                                                          rnn_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[src_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[src_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[weights_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[3], ctx->buffer_data[weights_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[4], ctx->buffer_data[bias_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[5], ctx->buffer_data[dst_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[6], ctx->buffer_data[dst_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[7], ctx->mkldnn_workspaces[deps[8]]);

                    cpu::mkldnn_utils::mkldnn_invoke_primitive(
                        ctx, rnn_index, deps, cpu::mkldnn_utils::OpType::RNN);
                };
                functors.emplace_back(functor);
#else
                mkldnn_emitter->query_scratchpad_rnn_forward(rnn_desc);

                auto src_iter_c_buffer_index =
                    external_function->get_buffer_index(args[2].get_name());
                auto weights_layer_buffer_index =
                    external_function->get_buffer_index(args[3].get_name());
                auto weights_iter_buffer_index =
                    external_function->get_buffer_index(args[4].get_name());
                auto bias_buffer_index = external_function->get_buffer_index(args[5].get_name());
                auto dst_iter_c_buffer_index =
                    external_function->get_buffer_index(out[2].get_name());

                // Rnn needs 11 primitives: src_layer, src_iter, src_iter_c, weights_layer,
                // weights_iter, bias,
                // dst_layer, dst_iter, dst_iter_c, workspace, and lstm_forward.
                // It needs a new workspace.
                auto rnn_index =
                    mkldnn_emitter->reserve_primitive_space(11, true /* new workspace */);
                auto& deps = mkldnn_emitter->get_primitive_deps(rnn_index);

                auto functor = [&,
                                rnn_desc,
                                rnn_index,
                                src_layer_buffer_index,
                                src_iter_buffer_index,
                                src_iter_c_buffer_index,
                                weights_layer_buffer_index,
                                weights_iter_buffer_index,
                                bias_buffer_index,
                                dst_layer_buffer_index,
                                dst_iter_buffer_index,
                                dst_iter_c_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* ectx) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_rnn_forward(ctx->mkldnn_memories,
                                                          ctx->mkldnn_primitives,
                                                          ctx->mkldnn_scratchpad_mds,
                                                          ctx->mkldnn_workspaces,
                                                          rnn_desc,
                                                          deps,
                                                          rnn_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[src_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[src_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[src_iter_c_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[3], ctx->buffer_data[weights_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[4], ctx->buffer_data[weights_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[5], ctx->buffer_data[bias_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[6], ctx->buffer_data[dst_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[7], ctx->buffer_data[dst_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[8], ctx->buffer_data[dst_iter_c_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[9], ctx->mkldnn_workspaces[deps[10]]);

                    cpu::mkldnn_utils::mkldnn_invoke_primitive(
                        ctx, rnn_index, deps, cpu::mkldnn_utils::OpType::RNN);
                };
                functors.emplace_back(functor);
#endif
            }

            void register_builders_rnn_cpp() { REGISTER_OP_BUILDER(Rnn); }
        }
    }
}
