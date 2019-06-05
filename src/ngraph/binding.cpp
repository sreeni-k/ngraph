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

#include "ngraph/binding.hpp"

using namespace ngraph;

std::shared_ptr<Binding> ngraph::bind(const ParameterVector& parameters,
                                      const OutputVector& arguments)
{
    return std::shared_ptr<Binding>(new Binding(parameters, arguments));
}

Binding::Binding(const ParameterVector& parameters, const OutputVector& arguments)
    : m_parameters(parameters)
    , m_arguments(arguments)
{
}

const ParameterVector& Binding::get_parameters() const
{
    return m_parameters;
}

const OutputVector& Binding::get_arguments() const
{
    return m_arguments;
}
