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

Binding::Binding()
{
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

ParameterVector& Binding::get_parameters()
{
    return m_parameters;
}

void Binding::set_parameters(const ParameterVector& parameters)
{
    m_parameters = parameters;
}

const OutputVector& Binding::get_arguments() const
{
    return m_arguments;
}

OutputVector& Binding::get_arguments()
{
    return m_arguments;
}

void Binding::set_arguments(const OutputVector& arguments)
{
    m_arguments = arguments;
}
