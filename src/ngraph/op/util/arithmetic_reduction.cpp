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

#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::util::ArithmeticReduction::ArithmeticReduction()
{
}

op::util::ArithmeticReduction::ArithmeticReduction(const Output<Node>& arg,
                                                   const AxisSet& reduction_axes)
    : Op({arg,
          op::Constant::create(
              element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())
              ->output(0)})
{
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
}

op::util::ArithmeticReduction::ArithmeticReduction(const Output<Node>& arg,
                                                   const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes})
{
}

bool op::util::ArithmeticReduction::reduction_axes_constant() const
{
    return is_type<op::Constant>(input_value(1).get_node());
}

const AxisSet op::util::ArithmeticReduction::get_reduction_axes() const
{
    AxisSet axes;
    if (auto const_op = as_type<op::Constant>(input_value(1).get_node()))
    {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

void op::util::ArithmeticReduction::set_reduction_axes(const AxisSet& reduction_axes)
{
    this->input(1).replace_source_output(
        op::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())
            ->output(0));
}

void op::util::ArithmeticReduction::validate_and_infer_types()
{
    auto reduction_axes = get_reduction_axes();
    auto input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    PartialShape result_shape{PartialShape::dynamic()};

    if (input_rank.is_static() && reduction_axes_constant())
    {
        std::vector<Dimension> dims;

        for (auto axis : reduction_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                  axis < size_t(input_rank),
                                  "Reduction axis (",
                                  axis,
                                  ") is out of bounds ",
                                  "(argument shape: ",
                                  input_shape,
                                  ", reduction axes: ",
                                  reduction_axes,
                                  ")");
        }

        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            if (reduction_axes.count(i) == 0)
            {
                dims.push_back(input_shape[i]);
            }
        }

        result_shape = PartialShape(dims);
    }

    set_input_is_relevant_to_shape(1);

    set_output_type(0, get_input_element_type(0), result_shape);
}
