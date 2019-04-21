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

#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::Executable::Executable()
{
}

runtime::Executable::~Executable()
{
}

bool runtime::Executable::call_with_validate(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                             const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    validate(outputs, inputs);
    return call(outputs, inputs);
}

void runtime::Executable::validate(const vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                   const vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_INFO << parameters.size();
    NGRAPH_INFO << static_cast<void*>(this);
    NGRAPH_INFO << outputs.size();
    NGRAPH_INFO << inputs.size();
    const ResultVector& results = get_results();
    NGRAPH_INFO << results.size();
    if (parameters.size() != inputs.size())
    {
        NGRAPH_INFO;
        stringstream ss;
        ss << "Call input count " << inputs.size() << " does not match Function's Parameter count "
           << parameters.size();
        throw runtime_error(ss.str());
    }
    NGRAPH_INFO;
    if (results.size() != outputs.size())
    {
        NGRAPH_INFO;
        stringstream ss;
        ss << "Call output count " << outputs.size() << " does not match Function's Result count "
           << results.size();
        throw runtime_error(ss.str());
    }

    NGRAPH_INFO;
    for (size_t i = 0; i < parameters.size(); i++)
    {
        NGRAPH_INFO;
        if (parameters[i]->get_element_type() != inputs[i]->get_element_type())
        {
            NGRAPH_INFO;
            stringstream ss;
            ss << "Input " << i << " type '" << inputs[i]->get_element_type()
               << "' does not match Parameter type '" << parameters[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        NGRAPH_INFO;
        if (parameters[i]->get_shape() != inputs[i]->get_shape())
        {
            NGRAPH_INFO;
            stringstream ss;
            ss << "Input " << i << " shape {" << join(inputs[i]->get_shape())
               << "} does not match Parameter shape {" << join(parameters[i]->get_shape()) << "}";
            throw runtime_error(ss.str());
        }
    }

    NGRAPH_INFO;
    for (size_t i = 0; i < results.size(); i++)
    {
        NGRAPH_INFO;
        if (results[i]->get_element_type() != outputs[i]->get_element_type())
        {
            NGRAPH_INFO;
            stringstream ss;
            ss << "Output " << i << " type '" << outputs[i]->get_element_type()
               << "' does not match Result type '" << results[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        NGRAPH_INFO;
        if (results[i]->get_shape() != outputs[i]->get_shape())
        {
            NGRAPH_INFO;
            stringstream ss;
            ss << "Output " << i << " shape {" << join(outputs[i]->get_shape())
               << "} does not match Result shape {" << join(results[i]->get_shape()) << "}";
            throw runtime_error(ss.str());
        }
    }
}

const ngraph::ParameterVector& runtime::Executable::get_parameters() const
{
    NGRAPH_INFO << static_cast<const void*>(this);
    NGRAPH_INFO << m_parameters.size();
    return m_parameters;
}

const ngraph::ResultVector& runtime::Executable::get_results() const
{
    return m_results;
}

void runtime::Executable::set_parameters_and_results(const Function& func)
{
    m_parameters = func.get_parameters();
    m_results = func.get_results();
    NGRAPH_INFO << m_parameters.size();
    NGRAPH_INFO << m_results.size();
    NGRAPH_INFO << static_cast<void*>(this);
}

vector<runtime::PerformanceCounter> runtime::Executable::get_performance_data() const
{
    return vector<PerformanceCounter>();
}
