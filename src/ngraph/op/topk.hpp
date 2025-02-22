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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        // \brief Computes indices of top k maximum/minimum index along a specified axis for a
        //        given tensor
        class TopK : public Op
        {
        public:
            enum class SortType
            {
                // Returned values are not sorted
                NONE,
                // Sort result based on element indices
                SORT_INDICES,
                // Sort result based on element values
                SORT_VALUES,
            };

            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"TopK", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            /// \brief Constructs a TopK operation
            TopK() = default;
            /// \brief Constructs a TopK operation.
            ///
            /// \param arg The input tensor
            /// \param top_k_axis The axis along which to compute top k indices
            /// \param index_element_type produce indices. Currently, only int64 or int32 are
            ///                           supported
            /// \param k Number of top indices to compute. Compute all indices if k = 0
            /// \param compute_max Compute top k max or top k min?
            /// \param sort SortType for sorting results, default - SORT_VALUES
            TopK(const Output<Node>& arg,
                 size_t top_k_axis,
                 const element::Type& index_element_type,
                 size_t k = 0,
                 bool compute_max = true,
                 SortType sort = SortType::SORT_VALUES);
            /// \brief Constructs a TopK operation.
            ///
            /// \param arg The input tensor
            /// \param k Number of top indices to compute. Compute all indices if k = 0
            /// \param top_k_axis The axis along which to compute top k indices
            /// \param index_element_type produce indices. Currently, only int64 or int32 are
            ///                           supported
            /// \param compute_max Compute top k max or top k min?
            /// \param sort SortType for sorting results, default - SORT_VALUES
            TopK(const Output<Node>& arg,
                 const Output<Node>& k,
                 size_t top_k_axis,
                 const element::Type& index_element_type,
                 bool compute_max = true,
                 SortType sort = SortType::SORT_VALUES);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_k() const;
            void set_k(size_t k);

            size_t get_top_k_axis() const { return m_top_k_axis; }
            element::Type get_index_element_type() const { return m_index_element_type; }
            bool get_compute_max() const { return m_compute_max; }
            SortType get_sort() const { return m_sort; }
        protected:
            size_t m_top_k_axis{0};
            element::Type m_index_element_type;
            bool m_compute_max{false};
            SortType m_sort{SortType::NONE};
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };

        namespace v1
        {
            /// \brief Computes indices and values of the k maximum/minimum values
            ///        for each slice along specified axis.
            class TopK : public Op
            {
            public:
                enum class SortType
                {
                    NONE,
                    SORT_INDICES,
                    SORT_VALUES,
                };

                enum class Mode
                {
                    MAX,
                    MIN
                };

                NGRAPH_API
                static constexpr NodeTypeInfo type_info{"TopK", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a TopK operation
                TopK() = default;
                /// \brief Constructs a TopK operation with two outputs: values and indices.
                ///        By default the indices output is described by i32 data type.
                ///
                /// \param data The input tensor
                /// \param k Specifies how many maximum/minimum elements should be computed
                ///          (note: scalar input tensor)
                /// \param axis The axis along which to compute top k indices
                /// \param mode Specifies which operation (min or max) is used to select
                ///             the biggest element of two.
                /// \param sort Specifies order of output elements and/or indices
                ///             Accepted values: none, index, value
                /// \param index_element_type Specyfies type of produced indices
                TopK(const Output<Node>& data,
                     const Output<Node>& k,
                     const int64_t axis,
                     const std::string& mode,
                     const std::string& sort,
                     const element::Type& index_element_type = element::i32);

                TopK(const Output<Node>& data,
                     const Output<Node>& k,
                     const int64_t axis,
                     const Mode mode,
                     const SortType sort,
                     const element::Type& index_element_type = element::i32);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                virtual size_t get_version() const override { return 1; }
                size_t get_axis() const { return m_axis; }
                void set_axis(const size_t axis) { m_axis = axis; }
                Mode get_mode() const { return m_mode; }
                void set_mode(const Mode mode) { m_mode = mode; }
                SortType get_sort_type() const { return m_sort; }
                void set_sort_type(const SortType sort) { m_sort = sort; }
                element::Type get_index_element_type() const { return m_index_element_type; }
                void set_index_element_type(const element::Type& index_element_type)
                {
                    m_index_element_type = index_element_type;
                }

                /// \brief Returns the value of K, if available
                ///
                /// \note If the second input to this op is a constant, the value is retrieved
                ///       and returned. If the input is not constant(dynamic) this method returns 0
                size_t get_k() const;
                void set_k(size_t k);

            protected:
                int64_t m_axis;
                Mode m_mode;
                SortType m_sort;
                element::Type m_index_element_type;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const NodeVector& deltas) override;

                size_t read_k_from_constant_node(const std::shared_ptr<Node>& node,
                                                 const element::Type& k_element_type) const;

                Mode mode_from_string(const std::string& mode) const;
                SortType sort_type_from_string(const std::string& sort) const;

                template <typename T>
                size_t validate_and_get_k(const std::shared_ptr<op::Constant>& k_constant) const;
            };
        }
    }
}
