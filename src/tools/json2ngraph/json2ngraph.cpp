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
                   << op->get_element_type().c_type_string() << ">{"
                   << join(constant->get_value_strings()) << "});\n";
        }
    }
    writer << "\n";

    for (const shared_ptr<Node>& op : f->get_ordered_ops())
    {
        if (op->description() != "Parameter" && op->description() != "Constant")
        {
            NGRAPH_INFO << op->get_name();
            writer << "auto " << to_variable(*op) << " = make_shared<op::" << op->description()
                   << ">(";
            vector<string> input_strs;
            for (auto input : op->inputs())
            {
                auto n = input.get_source_output().get_node();
                input_strs.push_back(to_variable(*n));
            }
            writer << join(input_strs);
            if (dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(op))
            {
            }
            else if (auto broadcast = dynamic_pointer_cast<op::Broadcast>(op))
            {
                writer << ", Shape{" << join(broadcast->get_broadcast_shape()) << "}, ";
                writer << "AxisSet{" << join(broadcast->get_broadcast_axes()) << "}";
            }
            else if (auto reshape = dynamic_pointer_cast<op::Reshape>(op))
            {
                writer << ", AxisVector{" << join(reshape->get_input_order()) << "}, ";
                writer << "Shape{" << join(reshape->get_output_shape()) << "}";
            }
            else if (auto result = dynamic_pointer_cast<op::Result>(op))
            {
            }
            writer << ");\n";
        }
    }

    writer.block_end();

    out << writer.get_code();
}
