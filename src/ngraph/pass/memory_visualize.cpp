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
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "memory_visualize.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
string number_string(T value)
{
    stringstream ss;
    ss.imbue(locale(""));
    ss << value;
    return ss.str();
}

pass::MemoryVisualize::MemoryVisualize(const string& filename)
    : m_filename{filename}
{
}

bool pass::MemoryVisualize::run_on_module(vector<shared_ptr<Function>>& functions)
{
    NGRAPH_INFO << m_filename;
    ofstream file(m_filename);
    {
        NGRAPH_INFO;
        for (shared_ptr<Function> f : functions)
        {
            NGRAPH_INFO;
            list<shared_ptr<Node>> nodes = f->get_ordered_ops();
            NGRAPH_INFO << nodes.size();
            file << "<!DOCTYPE html>\n<html>\n";
            file << "<head>\n";
            file << "    <style>\n";
            file << "        th, td {\n";
            file << "            border-bottom: 1px solid #ddd;\n";
            file << "            width: 200px;\n";
            file << "        }\n";
            file << "        table, td, th {\n";
            // file << "            border: 1px solid #ddd;\n";
            // file << "            text-align: left;\n";
            file << "        }\n";
            file << "        table {\n";
            file << "            border-collapse: collapse;\n";
            // file << "            width: 100%;\n";
            file << "        }\n";
            // file << "        tr:hover {background-color: #f5f5f5}\n";
            file << "        tr:nth-child(even) {background-color: #f2f2f2}\n";
            file << "    </style>\n";
            file << "</head>\n";

            file << "<body>\n";
            unordered_set<descriptor::Tensor*> tensors;
            size_t temp_max_size = 0;
            for (const shared_ptr<Node>& node : nodes)
            {
                tensors.insert(node->liveness_new_list.begin(), node->liveness_new_list.end());
            }
            NGRAPH_INFO << tensors.size();
            for (descriptor::Tensor* tensor : tensors)
            {
                temp_max_size += tensor->size();
            }

            file << "<table>\n";
            // file << "<tr><td>Persistent Memory Footprint</td><td align=\"right\">";
            // file << computation_decl.exop_block.persistent_size() << "</td></tr>\n";
            // file << "<tr><td>Temporary Memory Footprint</td><td align=\"right\">";
            // file << computation_decl.exop_block.memory_footprint() << "</td></tr>\n";
            file << "<tr><td>Max temporary Memory Footprint</td><td align=\"right\">";
            file << number_string(temp_max_size) << "</td></tr>\n";
            file << "</table>\n";

            // file << "<hr>\n";
            // draw_tensor_weight(file, nodes);
            // file << "<hr>\n";
            // draw_op_influence(file);
            file << "<hr>\n";
            draw_histogram(file, nodes);
            // file << "<hr>\n";
            file << "</body>\n</html>\n";
        }
    }
    return false;
}

unordered_set<const descriptor::Tensor*>
    pass::MemoryVisualize::find_largest_op(const list<shared_ptr<Node>>& nodes)
{
    size_t largest_size = 0;
    unordered_set<const descriptor::Tensor*> liveness_list;
    unordered_set<const descriptor::Tensor*> largest_live_list;
    for (shared_ptr<Node> exop : nodes)
    {
        size_t size = 0;
        for (const descriptor::Tensor* tensor : exop->liveness_new_list)
        {
            liveness_list.insert(tensor);
            size += tensor->size();
        }
        for (const descriptor::Tensor* tensor : liveness_list)
        {
            size += tensor->size();
        }
        if (size > largest_size)
        {
            largest_size = size;
            largest_live_list = liveness_list;
        }
    }
    return largest_live_list;
}

void pass::MemoryVisualize::draw_tensor_weight(ostream& file, const list<shared_ptr<Node>>& nodes)
{
    unordered_set<const descriptor::Tensor*> largest_live_list = find_largest_op(nodes);

    unordered_map<const descriptor::Tensor*, size_t> age_list;
    vector<const descriptor::Tensor*> tensor_set;
    file << "<table>\n";
    file << "    <tr>";
    file << "<th align=\"left\">tensor</th>";
    file << "<th align=\"right\">size</th>";
    file << "<th align=\"right\">age</th>";
    file << "</tr>\n";
    size_t i = 0;
    for (shared_ptr<Node> exop : nodes)
    {
        for (const descriptor::Tensor* tensor : exop->liveness_new_list)
        {
            age_list[tensor] = i;
        }
        for (const descriptor::Tensor* tensor : exop->liveness_free_list)
        {
            size_t start = age_list[tensor];
            age_list[tensor] = (i - start);
            tensor_set.push_back(tensor);
        }
        i++;
    }
    sort(tensor_set.begin(),
         tensor_set.end(),
         [](const descriptor::Tensor* t1, const descriptor::Tensor* t2) {
             return t1->size() < t2->size();
         });
    for (const descriptor::Tensor* tensor : tensor_set)
    {
        if (largest_live_list.find(tensor) != largest_live_list.end())
        {
            file << "    <tr style=\"background-color: #f0c0f0\">";
        }
        else
        {
            file << "    <tr>";
        }
        file << "<td>" << tensor->get_name() << "</td>";
        file << "<td align=\"right\">" << tensor->size() << "</td>";
        file << "<td align=\"right\">" << age_list[tensor] << "</td>";
        file << "</tr>\n";
    }

    file << "</table>\n";
}

void pass::MemoryVisualize::draw_histogram(ostream& file, const list<shared_ptr<Node>>& nodes)
{
    size_t stroke_width = 14;
    size_t text_offset = 4;
    size_t offset = 200;
    size_t width = 1000;
    size_t scale = width - offset;
    size_t line_spacing = static_cast<size_t>(stroke_width * 1.5);
    size_t line_count = nodes.size();
    size_t height = line_count * line_spacing + stroke_width;
    size_t memory_footprint = max<size_t>(1, MemoryVisualize::memory_footprint(nodes));

    file << "<svg viewBox=\"0 0 " << width << " " << height << "\">\n";
    size_t y = 0;
    unordered_set<const descriptor::Tensor*> liveness_list;
    size_t memory_size = 0;
    for (const shared_ptr<Node>& node : nodes)
    {
        for (const descriptor::Tensor* tensor : node->liveness_new_list)
        {
            liveness_list.insert(tensor);
            memory_size += tensor->size();
        }

        // float usage = float(MemoryVisualize::memory_usage(node));
        float usage = float(memory_size);
        float footprint = float(MemoryVisualize::memory_footprint(node));
        y += line_spacing;
        size_t x1 = offset;
        size_t x2 = static_cast<size_t>(((usage / memory_footprint) * scale) + offset);
        file << "<text x=\"" << 0 << "\" y=\"" << y + text_offset << "\" fill=\""
             << "black"
             << "\">" << node->get_name() << "</text>\n";
        file << "<line x1=\"" << x1 << "\" y1=\"" << y << "\" x2=\"" << x2 << "\" y2=\"" << y
             << "\"";
        file << " style=\"stroke:forestgreen;stroke-width:" << stroke_width << "\" />\n";
        x1 = x2;
        x2 = static_cast<size_t>(((footprint / memory_footprint) * scale) + offset);
        file << "<line x1=\"" << x1 << "\" y1=\"" << y << "\" x2=\"" << x2 << "\" y2=\"" << y
             << "\"";
        file << " style=\"stroke:firebrick;stroke-width:" << stroke_width << "\" />\n";

        for (const descriptor::Tensor* tensor : node->liveness_free_list)
        {
            liveness_list.erase(tensor);
            memory_size -= tensor->size();
        }
    }
    file << "</svg>\n";
}

size_t pass::MemoryVisualize::memory_usage(const shared_ptr<Node>& node)
{
    // ""
    // Get the memory usage of this op which is the sum of the sizes of all
    // off the live tensors.
    // Arguments:
    //   None
    // Returns:
    //   Memory usage in bytes
    // """
    // size = 0
    // for node in self.liveness_live_list:
    //     size += self.align(node.size)
    // return size

    return 0;
}

size_t pass::MemoryVisualize::memory_footprint(const shared_ptr<Node>& node)
{
    // """
    // Get the total memory footprint of this op. The footprint hightest memory
    // address used by the tensors in this op
    // Arguments:
    //   None
    // Returns:
    //   Memory footprint in bytes
    // """
    // max_mem = 0
    // for node in self.liveness_live_list:
    //     if node.buffer_pool_offset is not None:
    //         offset = self.align(node.size) + node.buffer_pool_offset
    //         max_mem = max([offset, max_mem])
    // return max_mem
    return 0;
}

size_t pass::MemoryVisualize::memory_footprint(const std::list<shared_ptr<Node>>& nodes)
{
    size_t largest_size = 0;
    size_t memory_size = 0;
    for (const shared_ptr<Node>& node : nodes)
    {
        for (const descriptor::Tensor* tensor : node->liveness_new_list)
        {
            memory_size += tensor->size();
        }
        largest_size = max<size_t>(largest_size, memory_size);
        for (const descriptor::Tensor* tensor : node->liveness_free_list)
        {
            memory_size -= tensor->size();
        }
    }
    return largest_size;
}
