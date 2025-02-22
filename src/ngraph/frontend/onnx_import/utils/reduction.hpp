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

#pragma once

#include <cstdint> // std::int64_t
#include <memory>  // std::make_shared
#include <utility> // std::move

#include "core/node.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            namespace detail
            {
                AxisSet get_reduction_axes(const Node& node);

            } // namespace  detail

            using ReductionFunction = std::function<std::shared_ptr<ngraph::Node>(
                const std::shared_ptr<ngraph::Node>&, const ngraph::AxisSet&)>;

            ///
            /// \brief      Create an nGraph version of an ONNX reduction operation.
            ///
            /// \param[in]  node                The node representing incoming ONNX operation.
            /// \param[in]  ng_input            The input (nGraph) Tensor.
            /// \param[in]  reduction_function  The reduction function defining arithmetic reduction
            ///                                 operation (e.g. Min, Max, Sum, Product).
            ///
            /// \return     nGraph node equivalent of the ONNX operation.
            ///
            std::shared_ptr<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const std::shared_ptr<ngraph::Node>& ng_input,
                                     ReductionFunction reduction_function);

            template <class IndexReduction>
            std::shared_ptr<ngraph::Node> make_ng_index_reduction_op(const Node& node)
            {
                auto axis = node.get_attribute_value<std::int64_t>("axis", 0);
                auto keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);
                auto input_node = node.get_ng_inputs().at(0);
                auto valid_axis = common::validate_axis(node, axis, input_node->get_shape().size());

                auto op_node =
                    std::make_shared<IndexReduction>(input_node, valid_axis, element::i64);

                if (keepdims == 0)
                {
                    return std::move(op_node);
                }

                // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                auto convert_node = std::make_shared<ngraph::op::Convert>(op_node, element::f32);

                auto output_shape = input_node->get_shape();
                output_shape.at(valid_axis) = 1;
                auto reshape_node = std::make_shared<ngraph::op::Reshape>(
                    convert_node,
                    ngraph::get_default_order(op_node->get_shape().size()),
                    Shape{output_shape});

                // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                auto reconvert_node =
                    std::make_shared<ngraph::op::Convert>(reshape_node, element::i64);

                return std::move(reconvert_node);
            }

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph
