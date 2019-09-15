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

#include "json2ngraph.hpp"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

static string to_variable(const Node& n)
{
    return to_lower(n.get_name());
}

void json2ngraph(istream& in, ostream& out)
{
    CodeWriter writer;
    writer.block_begin();
    shared_ptr<Function> f = deserialize(in);

    // Construct all shapes used in the model
    map<Shape, string> shape_names;
    for (const shared_ptr<Node>& op : f->get_ops())
    {
        if (op->description() == "Parameter" || op->description() == "Constant")
        {
            Shape shape = op->get_output_shape(0);
            if (shape_names.find(shape) == shape_names.end())
            {
                static size_t shape_index = 0;
                shape_names.insert({shape, "shape_" + to_string(shape_index++)});
            }
        }
    }

    // Output all of the Shape declarations use in the mode
    for (auto shape_info : shape_names)
    {
        writer << "Shape " << shape_info.second << "{" << join(shape_info.first) << "};\n";
    }
    writer << "\n";

    // Output all Parameter declarations
    for (const shared_ptr<op::Parameter>& parameter : f->get_parameters())
    {
        writer << "auto " << to_variable(*parameter) << " = make_shared<op::Parameter>(element::"
               << parameter->get_element_type().get_type_name() << ", "
               << shape_names[parameter->get_output_shape(0)] << ");\n";
    }
    writer << "\n";

    // Output all Constant declarations
    for (const shared_ptr<Node>& op : f->get_ops())
    {
        if (op->description() == "Constant")
        {
            auto constant = static_pointer_cast<op::Constant>(op);
            writer << "auto " << to_variable(*op) << " = make_shared<op::Constant>(element::"
                   << op->get_element_type().get_type_name() << ", "
                   << shape_names[op->get_output_shape(0)] << ", vector<"
                   << op->get_element_type().c_type_string() << ">{";
            if (constant->are_all_data_elements_bitwise_identical())
            {
                writer << constant->convert_value_to_string(0);
            }
            else
            {
                writer << join(constant->get_value_strings());
            }
            writer << "});\n";
        }
    }
    writer << "\n";

    for (const shared_ptr<Node>& op : f->get_ordered_ops())
    {
        if (op->description() != "Parameter" && op->description() != "Constant")
        {
            writer << "auto " << to_variable(*op) << " = make_shared<op::" << op->description()
                   << ">(";
            vector<string> input_strings;
            for (auto input : op->inputs())
            {
                auto n = input.get_source_output().get_node();
                input_strings.push_back(to_variable(*n));
            }
            if (dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(op) ||
                dynamic_pointer_cast<op::util::BinaryElementwiseComparison>(op) ||
                dynamic_pointer_cast<op::util::BinaryElementwiseLogical>(op))
            {
                writer << join(input_strings);
            }
            else if (dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(op))
            {
                writer << join(input_strings);
            }
            else if (auto cop = dynamic_pointer_cast<op::util::ArithmeticReduction>(op))
            {
                writer << join(input_strings);
                if (op->inputs().size() == 1)
                {
                    writer << ", " << cop->get_reduction_axes();
                }
            }
            else if (auto broadcast = dynamic_pointer_cast<op::Broadcast>(op))
            {
                writer << join(input_strings);
                writer << ", " << broadcast->get_broadcast_shape();
                writer << ", " << broadcast->get_broadcast_axes();
            }
            else if (auto reshape = dynamic_pointer_cast<op::Reshape>(op))
            {
                writer << join(input_strings);
                writer << ", " << reshape->get_input_order();
                writer << ", " << reshape->get_output_shape();
            }
            else if (auto result = dynamic_pointer_cast<op::Result>(op))
            {
                writer << join(input_strings);
            }
            else if (auto cop = dynamic_pointer_cast<op::Convolution>(op))
            {
                writer << join(input_strings);
                writer << ", " << cop->get_window_movement_strides();
                writer << ", " << cop->get_window_dilation_strides();
                writer << ", " << cop->get_padding_below();
                writer << ", " << cop->get_padding_above();
                writer << ", " << cop->get_data_dilation_strides();
                writer << ", op::PadType::" << cop->get_pad_type();
            }
            else if (auto cop = dynamic_pointer_cast<op::ConvolutionBackpropData>(op))
            {
                writer << cop->get_data_batch_shape();
                writer << ", " << join(input_strings);
                writer << ", " << cop->get_window_movement_strides_forward();
                writer << ", " << cop->get_window_dilation_strides_forward();
                writer << ", " << cop->get_padding_below_forward();
                writer << ", " << cop->get_padding_above_forward();
                writer << ", " << cop->get_data_dilation_strides_forward();
            }
            else if (auto cop = dynamic_pointer_cast<op::ConvolutionBackpropFilters>(op))
            {
                writer << input_strings[0];
                writer << ", " << cop->get_filters_shape();
                writer << ", " << input_strings[1];
                writer << ", " << cop->get_window_movement_strides_forward();
                writer << ", " << cop->get_window_dilation_strides_forward();
                writer << ", " << cop->get_padding_below_forward();
                writer << ", " << cop->get_padding_above_forward();
                writer << ", " << cop->get_data_dilation_strides_forward();
            }
            else if (auto bn = dynamic_pointer_cast<op::BatchNormTraining>(op))
            {
                writer << join(input_strings);
                writer << ", " << bn->get_eps_value();
            }
            else if (auto bn = dynamic_pointer_cast<op::BatchNormTrainingBackprop>(op))
            {
                writer << join(input_strings);
                writer << ", " << bn->get_eps_value();
            }
            else if (auto goe = dynamic_pointer_cast<op::GetOutputElement>(op))
            {
                writer << join(input_strings);
                writer << ", " << goe->get_n();
            }
            else if (auto ap = dynamic_pointer_cast<op::AvgPool>(op))
            {
                writer << join(input_strings);
                writer << ", " << ap->get_window_shape();
                writer << ", " << ap->get_window_movement_strides();
                writer << ", " << ap->get_padding_below();
                writer << ", " << ap->get_padding_above();
                writer << ", " << ap->get_include_padding_in_avg_computation();
                writer << ", op::PadType::" << ap->get_pad_type();
                writer << ", " << ap->get_ceil_mode();
            }
            else if (auto cop = dynamic_pointer_cast<op::AvgPoolBackprop>(op))
            {
                writer << cop->get_forward_arg_shape();
                writer << ", " << join(input_strings);
                writer << ", " << cop->get_window_shape();
                writer << ", " << cop->get_window_movement_strides();
                writer << ", " << cop->get_padding_below();
                writer << ", " << cop->get_padding_above();
                writer << ", " << cop->get_include_padding_in_avg_computation();
            }
            else if (auto convert = dynamic_pointer_cast<op::Convert>(op))
            {
                writer << join(input_strings);
                writer << ", element::" << convert->get_element_type().get_type_name();
            }
            else if (auto oh = dynamic_pointer_cast<op::OneHot>(op))
            {
                writer << join(input_strings);
                writer << ", PartialShape{" << join(oh->get_shape()) << "}";
                writer << ", " << oh->get_one_hot_axis();
            }
            else if (auto dot = dynamic_pointer_cast<op::Dot>(op))
            {
                writer << join(input_strings);
                writer << ", " << dot->get_reduction_axes_count();
                writer << ", " << dot->get_has_reduction_axes_count();
            }
            else if (auto cop = dynamic_pointer_cast<op::Softmax>(op))
            {
                writer << join(input_strings);
                writer << ", " << cop->get_axes();
            }
            else if (auto cop = dynamic_pointer_cast<op::Select>(op))
            {
                writer << join(input_strings);
            }
            else if (auto cop = dynamic_pointer_cast<op::TopK>(op))
            {
                writer << join(input_strings);
                writer << ", " << cop->get_top_k_axis();
                writer << ", element::" << cop->get_index_element_type().get_type_name();
                if (input_strings.size() == 1)
                {
                    writer << ", " << cop->get_k();
                }
                writer << ", " << cop->get_compute_max();
                switch (cop->get_sort())
                {
                case op::TopK::SortType::NONE: writer << ", op::TopK::SortType::NONE"; break;
                case op::TopK::SortType::SORT_INDICES:
                    writer << ", op::TopK::SortType::SORT_INDICES";
                    break;
                case op::TopK::SortType::SORT_VALUES:
                    writer << ", op::TopK::SortType::SORT_VALUES";
                    break;
                }
            }
            else
            {
                NGRAPH_INFO << writer.get_code();
                throw runtime_error("Unsupported op '" + op->description() + "'");
            }
            writer << ");\n";
        }
    }

    writer.block_end();

    out << writer.get_code();
}
