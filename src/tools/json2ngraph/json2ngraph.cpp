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
#include "ngraph/code_writer.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/recv.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/send.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

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
