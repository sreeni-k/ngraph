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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void ngraph::traverse_nodes(const std::shared_ptr<const Function> p,
                            std::function<void(std::shared_ptr<Node>)> f,
                            bool include_control_deps)
{
    traverse_nodes(p.get(), f, include_control_deps);
}

void ngraph::traverse_nodes(const Function* p,
                            std::function<void(std::shared_ptr<Node>)> f,
                            bool include_control_deps)
{
    NodeVector nodes;

    for (auto r : p->get_results())
    {
        nodes.push_back(r);
    }

    for (auto param : p->get_parameters())
    {
        nodes.push_back(param);
    }

    traverse_nodes(nodes, f, include_control_deps);
}

void ngraph::traverse_nodes(const NodeVector& subgraph_results,
                            std::function<void(std::shared_ptr<Node>)> f,
                            bool include_control_deps,
                            const NodeVector& subgraph_params)
{
    std::unordered_set<Node*> instances_seen;
    std::stack<Node*, std::vector<Node*>> stack;
    for (auto& node_ptr : subgraph_params)
    {
        instances_seen.insert(node_ptr.get());
    }
    for (auto& node_ptr : subgraph_results)
    {
        stack.push(node_ptr.get());
    }

    while (stack.size() > 0)
    {
        Node* n = stack.top();
        stack.pop();
        if (instances_seen.insert(n).second)
        {
            f(n->shared_from_this());
            for (auto input : n->inputs())
            {
                stack.push(input.get_source_output().get_node());
            }

            if (include_control_deps)
            {
                for (auto& cdep : n->get_control_dependencies())
                {
                    stack.push(cdep.get());
                }
            }
        }
    }
}

NodeVector ngraph::find_common_args(std::shared_ptr<Node> node1, std::shared_ptr<Node> node2)
{
    std::unordered_set<std::shared_ptr<Node>> node1_args;

    auto compute_node1_args = [&node1_args](const std::shared_ptr<Node> node) {
        node1_args.insert(node);
    };

    traverse_nodes({node1}, compute_node1_args, false, NodeVector{});

    std::unordered_set<std::shared_ptr<Node>> node2_args;

    auto compute_node2_args = [&node2_args](const std::shared_ptr<Node> node) {
        node2_args.insert(node);
    };

    traverse_nodes({node2}, compute_node2_args, false, NodeVector{});

    NodeVector common_args;
    for (auto e : node1_args)
    {
        if (node2_args.count(e) > 0)
        {
            common_args.push_back(e);
        }
    }

    return common_args;
}

void ngraph::replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement)
{
    if (target->is_output())
    {
        throw ngraph_error("Result nodes cannot be replaced.");
    }

    NGRAPH_CHECK(!target->get_users().empty(),
                 "Attempted to replace unreachable node '",
                 *target,
                 "'. Replacement: '",
                 *replacement,
                 "'");

    // Fix input/output descriptors
    NGRAPH_CHECK(target->get_output_size() == replacement->get_output_size());

    if (ngraph::get_provenance_enabled())
    {
        auto common_args = ngraph::find_common_args(target, replacement);

        std::set<string> removed_subgraph_tags;

        auto set_replacement_prov = [&removed_subgraph_tags](std::shared_ptr<Node> node) {
            for (auto tag : node->get_provenance_tags())
            {
                removed_subgraph_tags.insert(tag);
            }
        };

        traverse_nodes({target}, set_replacement_prov, false, common_args);
        replacement->add_provenance_tags(removed_subgraph_tags);

        auto set_prov_new_nodes = [&removed_subgraph_tags](std::shared_ptr<Node> node) {
            node->add_provenance_tags(removed_subgraph_tags);
        };

        traverse_nodes({replacement}, set_prov_new_nodes, false, common_args);
    }

    // For each of target's output O with replacement output O_rep:
    //     For each O's connected downstream input I:
    //         Change I's connected upstream output to O_rep
    for (size_t i = 0; i < target->get_output_size(); i++)
    {
        for (auto& input : target->output(i).get_target_inputs())
        {
            input.replace_source_output(replacement->output(i));
        }
    }

    replacement->add_node_control_dependents(target);
    target->clear_control_dependents();
}

void ngraph::replace_nodes(
    const std::shared_ptr<Function>& f,
    const unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Parameter>>&
        parameter_replacement_map,
    const unordered_map<shared_ptr<Node>, shared_ptr<Node>>& body_replacement_map)
{
    auto& params = f->get_parameters();

    for (size_t i = 0; i < params.size(); i++)
    {
        if (parameter_replacement_map.count(params[i]) != 0 &&
            parameter_replacement_map.at(params[i]) != params[i])
        {
            f->replace_parameter(i, parameter_replacement_map.at(params[i]));
        }
    }

    for (auto& kv : body_replacement_map)
    {
        auto& k = kv.first;
        auto& v = kv.second;

        if (k != v)
        {
            f->replace_node(k, v);
        }
    }
}

// Check if all paths from X to a result go through Y
bool ngraph::is_post_dominated(Node* X, Node* Y)
{
    std::unordered_set<Node*> visited;
    std::stack<Node*, std::vector<Node*>> stack;
    stack.push(X);

    while (stack.size() > 0)
    {
        ngraph::Node* curr = stack.top();
        visited.insert(curr);
        if (curr->is_output())
        {
            return false;
        }
        stack.pop();
        if (curr != Y)
        {
            for (const auto& next : curr->get_users())
            {
                if (visited.count(next.get()) == 0)
                {
                    stack.push(next.get());
                }
            }
        }
    }
    return true;
}

std::list<std::shared_ptr<ngraph::Node>>
    ngraph::clone_nodes(const std::list<std::shared_ptr<ngraph::Node>>& nodes, NodeMap& node_map)
{
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes, true);
    for (auto node : sorted_nodes)
    {
        if (node_map.count(node.get()) == 0)
        {
            // get (already) cloned arguments and clone the node
            OutputVector cloned_args;
            for (auto input : node->inputs())
            {
                Output<Node> output = input.get_source_output();
                cloned_args.push_back(output.for_node(node_map.at(output.get_node())));
            }
            std::vector<std::shared_ptr<Node>> cloned_dependencies;
            for (auto& dependency : node->get_control_dependencies())
            {
                shared_ptr<Node>& dependent = node_map.at(dependency.get());
                if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) ==
                    cloned_dependencies.end())
                {
                    cloned_dependencies.push_back(dependent);
                }
            }
            auto cloned_node = node->copy_with_new_inputs(cloned_args, cloned_dependencies);
            if (node->get_friendly_name() != node->get_name())
            {
                // There is a friendly name for this node so copy it
                cloned_node->set_friendly_name(node->get_friendly_name());
            }

            for (auto tag : node->get_provenance_tags())
            {
                cloned_node->add_provenance_tag(tag);
            }
            node_map[node.get()] = cloned_node;
        }
    }

    // create and return list of cloned nodes
    // order matches input list (not necessarily topological)
    std::list<std::shared_ptr<ngraph::Node>> cloned_nodes;
    for (auto node : nodes)
    {
        cloned_nodes.push_back(node_map.at(node.get()));
    }
    return cloned_nodes;
}

std::shared_ptr<ngraph::Function> ngraph::clone_function(const ngraph::Function& func)
{
    NodeMap nm;
    return clone_function(func, nm);
}

std::shared_ptr<ngraph::Function> ngraph::clone_function(const ngraph::Function& func,
                                                         NodeMap& node_map)
{
    // clone function operations
    clone_nodes(func.get_ops(true), node_map);

    // get cloned function results and parameters
    ResultVector cloned_results;
    for (shared_ptr<Node> node : func.get_results())
    {
        auto result = as_type_ptr<op::Result>(node_map.at(node.get()));
        if (!result)
        {
            throw ngraph_error("Results should be of type op::Result");
        }
        cloned_results.push_back(result);
    }
    std::vector<std::shared_ptr<op::Parameter>> cloned_params;
    for (auto param : func.get_parameters())
    {
        cloned_params.push_back(as_type_ptr<op::Parameter>(node_map.at(param.get())));
    }

    // create and return cloned function
    return std::make_shared<ngraph::Function>(cloned_results, cloned_params);
}

bool ngraph::is_equal_to_const_value(std::string const_value, const Output<Node>& reduce_constant)
{
    if (auto rc = as_type_ptr<ngraph::op::Constant>(reduce_constant.get_node_shared_ptr()))
    {
        auto cshape = rc->get_shape();
        size_t n = shape_size(cshape);
        // way to construct a constant of a given type, shape, value
        std::vector<std::string> vector_zero{n, const_value};
        auto constant_val_op =
            std::make_shared<ngraph::op::Constant>(rc->get_element_type(), cshape, vector_zero);

        // way to compare elements to const_value
        size_t n_bytes = n * rc->get_element_type().size();
        NGRAPH_DEBUG << "Comparing " << n_bytes << " bytes";
        return !memcmp(constant_val_op->get_data_ptr(), rc->get_data_ptr(), n_bytes);
    }
    else
    {
        return false;
    }
}

// Insert result and parameter node between src_node and dst_node by splitting the graph
//
// Before:                        |  After:
// (Device:0)         (Device:1)  |  (Device:0)         (Device:0)  (Device:1)         (Device:1)
// +-----+---+       +---+-----+  |  +-----+---+       +---+-----+  +-----+---+       +---+-----+
// |     |   |       |   |     |  |  |     |   |       |   |     |  |     |   |       |   |     |
// |     | o +--[0]--> i |     |  |  |     | o +--[4]--> i |     |  |     | o +--[8]--> i |     |
// |     |   <--[1]--+   |     |  |  |     |   <--[5]--+   |     |  |     |   <--[9]--+   |     |
// | src +---+       +---+ dst |  |  | src +---+       +---+ res |  | par +---+       +---+ dst |
// |     |               |     |  |  |     |               |     |  |     |               |     |
// |     +------[2]------>     |  |  |     +------[6]------>     |  |     +------[10]----->     |
// |     <------[3]------+     |  |  |     <------[7]------+     |  |     <------[11]-----+     |
// +-----+               +-----+  |  +-----+               +-----+  +-----+               +-----+
pair<shared_ptr<op::Result>, shared_ptr<op::Parameter>>
    ngraph::insert_result_parameter_split(const shared_ptr<Node>& src_node,
                                          const shared_ptr<Node>& dst_node)
{
    if (src_node->get_output_size() != 1)
    {
        throw ngraph_error("Multiple output per op not supported in graph partition yet.");
    }

    // Make parameter node
    shared_ptr<op::Parameter> par_node = make_shared<op::Parameter>(
        src_node->get_output_element_type(0), src_node->get_output_shape(0));
    par_node->set_placement(dst_node->get_placement());

    // Fix input / output among src, dst and par
    std::vector<Input<Node>> dst_inputs = get_inputs_from(*src_node, *dst_node);
    NGRAPH_CHECK(dst_inputs.size() == 1,
                 "insert_result_parameter_split encountered more than "
                 "one input between the source and destination nodes");
    auto& dst_input = dst_inputs[0];

    std::vector<Output<Node>> src_outputs = get_outputs_to(*src_node, *dst_node);
    NGRAPH_CHECK(src_outputs.size() == 1,
                 "insert_result_parameter_split encountered more than "
                 "one output between the source and destination nodes");
    auto& src_output = src_outputs[0];

    // Remove [0]
    src_output.remove_target_input(dst_input);

    // Remove [0] (again), add [8], remove [1], add [9]
    dst_input.replace_source_output(par_node->output(0));

    // Add res node
    // Add [4], [5], [6], [7]
    shared_ptr<op::Result> res_node = make_shared<op::Result>(src_node);
    res_node->set_placement(src_node->get_placement());

    return make_pair(res_node, par_node);
}

// Insert unary node between two nodes like S->D => S->N->D
// Before:                        |  After:
// +-----+---+       +---+-----+  |  +-----+---+       +---+-----+---+       +---+-----+
// |     |   |       |   |     |  |  |     |   |       |   |     |   |       |   |     |
// |     | o +--[0]--> i |     |  |  |     | o +--[4]--> i |     | o +--[8]--> i |     |
// |     |   <--[1]--+   |     |  |  |     |   <--[5]--+   |     |   <--[9]--+   |     |
// | src +---+       +---+ dst |  |  | src +---+       +---+ new +---+       +---+ dst |
// |     |               |     |  |  |     |               |     |               |     |
// |     +------[2]------>     |  |  |     +------[6]------>     +------[10]----->     |
// |     <------[3]------+     |  |  |     <------[7]------+     <------[11]-----+     |
// +-----+               +-----+  |  +-----+               +-----+               +-----+
//                                |
// +-----+---+       +---+-----+  |
// |     |   |       |   |     |  |
// |     | o +--[4]--> i |     |  |
// |     |   <--[5]--+   |     |  |
// | src +---+       +---+ new |  |
// |     |               |     |  |
// |     +------[6]------>     |  |
// |     <------[7]------+     |  |
// +-----+               +-----+  |
//
// This cannot be achieved by ngraph::replace_node().
// With replace_node(), we could do:
// [     S           S      ]
// [    / \          |      ]
// [   /   \   =>    N      ]
// [  /     \       / \     ]
// [ D0     D1    D0   D1   ]
//
// But we want:
// [     S            S     ]
// [    / \          / \    ]
// [   /   \   =>   N0  N1  ]
// [  /     \      /     \  ]
// [ D0     D1    D0     D1 ]
//
// Typically new_node is connected to src_node already. The reason we don't create `new_node`
// inside the function and return it (similar to ngraph::insert_result_parameter_split) is that
// we'll have to templatize its function to call new_node's constructor.
void ngraph::insert_new_node_between(const shared_ptr<Node>& src_node,
                                     const shared_ptr<Node>& dst_node,
                                     const shared_ptr<Node>& new_node)
{
    // Fix input / output
    std::vector<Input<Node>> dst_inputs = get_inputs_from(*src_node, *dst_node);
    NGRAPH_CHECK(dst_inputs.size() == 1,
                 "insert_new_node_between encountered more than one "
                 "input between the source and destination nodes");
    auto& dst_input = dst_inputs[0];

    std::vector<Output<Node>> src_outputs = get_outputs_to(*src_node, *dst_node);
    NGRAPH_CHECK(src_outputs.size() == 1,
                 "insert_new_node_between encountered more than one "
                 "output between the source and destination nodes");
    auto& src_output = src_outputs[0];

    src_output.remove_target_input(dst_input); // Remove [0]
    dst_input.replace_source_output(
        new_node->output(0)); // Remove [0] (again), add [8], remove [1], add [9]
}

std::shared_ptr<Node> ngraph::make_zero(const element::Type& element_type, const Shape& shape)
{
    std::shared_ptr<Node> zero = op::Constant::create(element_type, Shape{}, {0.0});
    if (shape.size() > 0)
    {
        AxisSet axes;
        for (size_t i = 0; i < shape.size(); i++)
        {
            axes.insert(i);
        }
        zero = std::make_shared<op::Broadcast>(zero, shape, axes);
    }
    return zero;
}

std::shared_ptr<Node> ngraph::make_constant_from_string(std::string val,
                                                        const element::Type& element_type,
                                                        const Shape& shape)
{
    auto cvals = std::vector<std::string>(shape_size(shape), val);
    return std::make_shared<op::Constant>(element_type, shape, cvals);
}

bool ngraph::is_zero(const Output<Node>& reduce_constant)
{
    auto result_bool = is_equal_to_const_value("0", reduce_constant);
    return result_bool;
}

bool ngraph::is_one(std::shared_ptr<Node> reduce_constant)
{
    auto result_bool = is_equal_to_const_value("1", reduce_constant);
    return result_bool;
}

NodeVector ngraph::get_subgraph_outputs(const NodeVector& nodes,
                                        const NodeVector& exclusions,
                                        bool ignore_unused,
                                        bool ignore_output_duplicates)
{
    std::set<shared_ptr<Node>> exclusions_set(exclusions.begin(), exclusions.end());
    std::set<shared_ptr<Node>> nodes_set(nodes.begin(), nodes.end());

    NodeVector outputs;

    for (auto n : nodes)
    {
        if (exclusions_set.count(n) != 0)
        {
            continue;
        }

        for (const auto& u : n->get_users())
        {
            bool add_output = nodes_set.count(u) == 0 && (!ignore_unused || is_used(u.get()));
            // check if output is already captured
            add_output &= (ignore_output_duplicates ||
                           std::find(outputs.begin(), outputs.end(), n) == outputs.end());
            if (add_output)
            {
                outputs.push_back(n);
            }
        }
    }
    return outputs;
}

NodeVector ngraph::extract_subgraph(const NodeVector& results, const NodeVector& args)
{
    NodeVector subgraph;
    traverse_nodes(results, [&](std::shared_ptr<Node> n) { subgraph.push_back(n); }, true, args);
    return subgraph;
}

bool ngraph::is_used(Node* node)
{
    std::unordered_set<Node*> instances_seen;
    std::stack<Node*, std::vector<Node*>> stack;
    stack.push(node);

    while (stack.size() > 0)
    {
        ngraph::Node* n = stack.top();
        if (instances_seen.count(n) == 0)
        {
            if (n->is_output())
            {
                return true;
            }
            instances_seen.insert(n);
        }
        stack.pop();
        for (const auto& arg : n->get_users())
        {
            if (instances_seen.count(arg.get()) == 0)
            {
                stack.push(arg.get());
            }
        }
    }
    return false;
}

size_t ngraph::get_user_count(Node* node)
{
    size_t count = 0;
    for (const auto& node_user : node->get_users())
    {
        count += is_used(node_user.get());
    }
    return count;
}

bool ngraph::possibly_overwritten(Node* node)
{
    for (auto& output : node->outputs())
    {
        for (auto& input : output.get_target_inputs())
        {
            if (input.get_node()->is_op())
            {
                auto op = static_cast<ngraph::op::Op*>(input.get_node());
                if (auto op_annotations = op->get_op_annotations())
                {
                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                    {
                        if (input.get_index() == oi_pair.input && oi_pair.destructive)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool ngraph::is_strided(const Strides& strides)
{
    return std::any_of(strides.begin(), strides.end(), [](size_t stride) { return stride != 1; });
}

bool ngraph::is_valid_rank(const std::shared_ptr<Node>& node, std::vector<size_t> valid_ranks)
{
    auto node_rank = node->get_shape().size();
    for (auto rank : valid_ranks)
    {
        if (rank == node_rank)
        {
            return true;
        }
    }
    return false;
}

bool ngraph::compare_constants(const std::shared_ptr<Node>& n1, const std::shared_ptr<Node>& n2)
{
    if (!(n1->is_constant() && n2->is_constant()))
    {
        return false;
    }

    if (static_pointer_cast<op::Constant>(n1)->get_value_strings() !=
        static_pointer_cast<op::Constant>(n2)->get_value_strings())
    {
        return false;
    }

    return true;
}

void ngraph::plot_graph(
    std::shared_ptr<Function> f,
    const std::string& filename,
    std::function<void(const Node& node, std::vector<std::string>& attributes)> attributes)
{
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::VisualizeTree>(filename, attributes);
    pass_manager.run_passes(f);
}

std::vector<Input<Node>> ngraph::get_inputs_from(Node& src, Node& dst)
{
    std::vector<Input<Node>> result;

    for (auto& input : dst.inputs())
    {
        if (input.get_source_output().get_node() == &src)
        {
            result.push_back(input);
        }
    }

    return result;
}

std::vector<Output<Node>> ngraph::get_outputs_to(Node& src, Node& dst)
{
    std::vector<Output<Node>> result;

    for (auto& output : src.outputs())
    {
        bool targets_dst = false;

        for (auto& input : output.get_target_inputs())
        {
            if (input.get_node() == &dst)
            {
                targets_dst = true;
                break;
            }
        }

        if (targets_dst)
        {
            result.push_back(output);
        }
    }

    return result;
}

static bool check_for_cycles_bkwd(std::shared_ptr<ngraph::Node> node,
                                  std::deque<std::shared_ptr<ngraph::Node>>& path,
                                  std::unordered_set<std::shared_ptr<ngraph::Node>>& path_set,
                                  ngraph::NodeVector& cycle_nodes)
{
    path.push_back(node);
    path_set.insert(node);
    for (auto& input : node->inputs())
    {
        auto arg = input.get_source_output().get_node_shared_ptr();
        if (path_set.find(arg) != path_set.end())
        {
            for (auto it : path)
            {
                cycle_nodes.push_back(it);
            }
            // last node
            cycle_nodes.push_back(arg);
            return true;
        }
        if (check_for_cycles_bkwd(arg, path, path_set, cycle_nodes))
        {
            return true;
        }
    }
    path_set.erase(path.back());
    path.pop_back();
    return false;
}

static bool check_for_cycles_fwd(std::shared_ptr<ngraph::Node> node,
                                 std::deque<std::shared_ptr<ngraph::Node>>& path,
                                 std::unordered_set<std::shared_ptr<ngraph::Node>>& path_set,
                                 ngraph::NodeVector& cycle_nodes)
{
    path.push_back(node);
    path_set.insert(node);
    for (auto& arg : node->get_users())
    {
        if (path_set.find(arg) != path_set.end())
        {
            for (auto it : path)
            {
                cycle_nodes.push_back(it);
            }
            // last node
            cycle_nodes.push_back(arg);
            return true;
        }
        if (check_for_cycles_fwd(arg, path, path_set, cycle_nodes))
        {
            return true;
        }
    }
    path_set.erase(path.back());
    path.pop_back();
    return false;
}

bool ngraph::check_for_cycles(const ngraph::Function* func,
                              ngraph::NodeVector& cycle_nodes,
                              bool& is_bkwd_cycle)
{
    for (auto res : func->get_results())
    {
        std::deque<std::shared_ptr<Node>> path;
        // mirror of path stack for faster cycle check
        std::unordered_set<std::shared_ptr<Node>> path_set;
        if (check_for_cycles_bkwd(res, path, path_set, cycle_nodes))
        {
            is_bkwd_cycle = true;
            return true;
        }
    }

    for (auto param : func->get_parameters())
    {
        std::deque<std::shared_ptr<Node>> path;
        // mirror of path stack for faster cycle check
        std::unordered_set<std::shared_ptr<Node>> path_set;
        if (check_for_cycles_fwd(param, path, path_set, cycle_nodes))
        {
            is_bkwd_cycle = false;
            return true;
        }
    }
    // no cycles
    return false;
}
