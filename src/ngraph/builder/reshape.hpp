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

#include <cstddef>
#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief      Change shape of input tensor.
        ///
        /// \param[in]  node   The node producing the tensor to be reshaped.
        /// \param[in]  shape  The new shape for input tensor.
        ///
        /// \return     The node representing a Reshape operation.
        ///
        Output<Node> reshape(const Output<Node>& node, const Shape& shape);
        std::shared_ptr<Node> reshape(const std::shared_ptr<Node>& node, const Shape& shape)
        {
            return reshape(Output<Node>(node), shape).get_node_shared_ptr();
        }

        /// \brief Permute axes according to specified axes_order parameter.
        ///
        /// \param node The node which axes we want to permute.
        /// \param axes_order The permutation of node tensor axes.
        ///
        /// \return: New node with permuted axes.
        Output<Node> reorder_axes(const Output<Node>& node,
                                  std::vector<std::size_t> axes_order = {});
        std::shared_ptr<Node> reorder_axes(const std::shared_ptr<Node>& node,
                                           std::vector<std::size_t> axes_order = {})
        {
            return reorder_axes(Output<Node>(node), axes_order).get_node_shared_ptr();
        }

        /// \brief Return transposed tensor (with axes in reversed order).
        ///
        /// \param node Input tensor we want to transpose
        ///
        /// \return: New node with reversed dimensions.
        Output<Node> transpose(const Output<Node>& node);
        std::shared_ptr<Node> transpose(const std::shared_ptr<Node>& node)
        {
            return transpose(Output<Node>(node)).get_node_shared_ptr();
        }

        /// \brief Flatten the input tensor into a 2D matrix.
        ///
        /// \param node The tensor to be flattened.
        /// \param axis The axis dividing shape.
        ///
        /// \return The new node will be a 2D matrix representing the flattened input node.
        Output<Node> flatten(const Output<Node>& node, int axis);
        std::shared_ptr<Node> flatten(const std::shared_ptr<Node>& node, int axis)
        {
            return flatten(Output<Node>(node), axis).get_node_shared_ptr();
        }
    } // namespace  builder
} // namespace  ngraph
