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

#include "ngraph/pass/validate_graph.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

bool pass::ValidateGraph::run_on_module(vector<shared_ptr<Function>>& functions)
{
    for (shared_ptr<Function> f : functions)
    {
        validate_parameters(*f);
    }
    return false;
}

void pass::ValidateGraph::validate_parameters(const Function& function)
{
    auto parameters = function.get_parameters();
    auto op_list = function.get_ops();
    for (auto node : op_list)
    {
        shared_ptr<op::Parameter> p = dynamic_pointer_cast<op::Parameter>(node);
        if (nullptr != p)
        {
            auto it = find_if(parameters.begin(),
                              parameters.end(),
                              [p](shared_ptr<op::Parameter> q) { return (p == q); });
            if (it == parameters.end())
            {
                throw ngraph_error("Function references undeclared parameter");
            }
        }
    }

    // Check that all nodes user's are in the current Function
    unordered_set<Node*> node_set;
    for (const shared_ptr<Node>& n : op_list)
    {
        node_set.insert(n.get());
    }
    for (const shared_ptr<Node>& node : op_list)
    {
        for (const shared_ptr<Node>& user : node->get_users())
        {
            if (node_set.find(user.get()) == node_set.end())
            {
                stringstream ss;
                ss << "Node " << node->get_name() << " user " << user->get_name()
                   << " not in Function";
                throw ngraph_error(ss);
            }
        }
    }
}
