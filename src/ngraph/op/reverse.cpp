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

#include <algorithm>
#include <sstream>

#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reverse.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Reverse::type_info;

op::Reverse::Reverse(const Output<Node>& arg, const AxisSet& reversed_axes)
    : Op({arg})
    , m_reversed_axes(reversed_axes)
{
    constructor_validate_and_infer_types();
}

void op::Reverse::validate_and_infer_types()
{
    const auto input_shape = get_input_partial_shape(0);
    const Dimension input_rank = input_shape.rank();

    if (input_rank.is_static())
    {
        // Make sure all reversed axis indices are valid.
        for (size_t axis : m_reversed_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                  axis < size_t(input_rank),
                                  "Reverse axis (",
                                  axis,
                                  ") is out of bounds (argument shape: ",
                                  input_shape,
                                  ").");
        }
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::Reverse::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Reverse>(new_args.at(0), m_reversed_axes);
}

void op::Reverse::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    adjoints.add_delta(x, make_shared<op::Reverse>(delta, m_reversed_axes));
}

constexpr NodeTypeInfo op::v1::Reverse::type_info;

op::v1::Reverse::Reverse(const Output<Node>& data,
                         const Output<Node>& reversed_axes,
                         const std::string& mode)
    : Op({data, reversed_axes})
    , m_mode{mode_from_string(mode)}
{
    constructor_validate_and_infer_types();
}

op::v1::Reverse::Reverse(const Output<Node>& data,
                         const Output<Node>& reversed_axes,
                         const Mode mode)
    : Op({data, reversed_axes})
    , m_mode{mode}
{
    constructor_validate_and_infer_types();
}

void op::v1::Reverse::validate_and_infer_types()
{
    if (m_mode == Mode::MASK)
    {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(1) == element::boolean,
                              "In 'mask' mode the second input must contain boolean values.");
    }

    const auto input_shape = get_input_partial_shape(0);
    const auto input_rank = input_shape.rank();

    const auto rev_axes_shape = get_input_partial_shape(1);
    const auto rev_axes_rank = rev_axes_shape.rank();

    if (rev_axes_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<size_t>(rev_axes_rank) == 1,
                              "The reversed_axes input must be a 1D tensor (got ",
                              static_cast<size_t>(rev_axes_rank),
                              ").");

        if (m_mode == Mode::MASK)
        {
            if (input_rank.is_static() && rev_axes_shape[0].is_static())
            {
                const auto rev_axes_mask_elems_count = static_cast<size_t>(rev_axes_shape[0]);
                NODE_VALIDATION_CHECK(this,
                                      rev_axes_mask_elems_count == static_cast<size_t>(input_rank),
                                      "The number of elements in the reversed_axes tensor (",
                                      rev_axes_mask_elems_count,
                                      ") must match the input data tensor rank (",
                                      static_cast<size_t>(input_rank),
                                      ") in 'mask' mode.");
            }
        }
    }

    if (input_rank.is_static())
    {
        const auto rank = static_cast<size_t>(input_rank);
        const auto rev_axes_node = input_value(1).get_node_shared_ptr();

        if (rev_axes_node->is_constant())
        {
            const auto rev_axes_constant = dynamic_pointer_cast<op::Constant>(rev_axes_node);

            if (m_mode == Mode::INDEX)
            {
                const AxisSet rev_axes = rev_axes_constant->get_axis_set_val();

                NODE_VALIDATION_CHECK(this,
                                      rev_axes.size() <= rank,
                                      "Too many axes(",
                                      rev_axes,
                                      ") have been provided for given input shape(",
                                      input_shape,
                                      ").");

                bool all_axes_in_range = all_of(rev_axes.begin(),
                                                rev_axes.end(),
                                                [&rank](const size_t axis) { return axis < rank; });

                NODE_VALIDATION_CHECK(this,
                                      all_axes_in_range,
                                      "Some of the provided axes (",
                                      rev_axes,
                                      ") are out of bounds (input rank: ",
                                      static_cast<size_t>(input_rank),
                                      ").");
            }
        }
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::v1::Reverse::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Reverse>(new_args.at(0), new_args.at(1), m_mode);
}

void op::v1::Reverse::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    const auto delta = deltas.at(0);

    const auto x = input_value(0);
    const auto reversed_axes = input_value(1);

    adjoints.add_delta(x, make_shared<op::v1::Reverse>(delta, reversed_axes, m_mode));
}

op::v1::Reverse::Mode op::v1::Reverse::mode_from_string(const std::string& mode) const
{
    static const std::map<std::string, Mode> allowed_values = {{"index", Mode::INDEX},
                                                               {"mask", Mode::MASK}};

    NODE_VALIDATION_CHECK(this, allowed_values.count(mode) > 0, "Invalid 'mode' value passed in.");

    return allowed_values.at(mode);
}
