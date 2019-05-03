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

#include <memory>

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        class AddKind : public NodeKind
        {
            friend class Add;
            AddKind()
                : NodeKind("Add")
            {
            }

        public:
            virtual std::shared_ptr<Node> make_shared(const NodeVector& args) override;
        };

        /// \brief Elementwise addition operation.
        ///
        class Add : public util::BinaryElementwiseArithmetic
        {
        public:
            static const AddKind node_kind;

            virtual const NodeKind* get_node_kind() override { return &node_kind; }
            /// \brief Constructs an addition operation.
            ///
            /// \param arg0 Node that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param arg1 Node that produces the second input tensor.<br>
            /// `[d0, ...]`
            ///
            /// Output `[d0, ...]`
            ///
            Add(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            virtual bool is_commutative() override { return true; }
        };
    } // namespace op

    std::shared_ptr<ngraph::Node> operator+(const std::shared_ptr<ngraph::Node> arg0,
                                            const std::shared_ptr<ngraph::Node> arg1);
} // namespace ngraph
