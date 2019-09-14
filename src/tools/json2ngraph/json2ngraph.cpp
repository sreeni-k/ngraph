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
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/code_writer.hpp"

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
    for (const shared_ptr<op::Parameter>& parameter : f->get_parameters())
    {
        Shape shape = parameter->get_output_shape(0);
        if (shape_names.find(shape) == shape_names.end())
        {
            static size_t shape_index = 0;
            shape_names.insert({shape, "shape_"+to_string(shape_index++)});
        }
    }

    for (auto shape_info : shape_names)
    {
        writer << "Shape " << shape_info.second << "{" << join(shape_info.first)<< "};\n";
    }

    writer.block_end(); 

    out << writer.get_code();
}
