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

#pragma once

#include "gtest/gtest.h"

#ifndef MSE_DEFAULT_MEAN
#define MSE_DEFAULT_MEAN 0
#endif
#ifndef MSE_DEFAULT_VARIANCE
#define MSE_DEFAULT_VARIANCE 0
#endif

namespace ngraph
{
    namespace test
    {
        template <typename T>
        ::testing::AssertionResult mse(const std::vector<T>& actual,
                                       const std::vector<T>& expected,
                                       float mean = MSE_DEFAULT_MEAN,
                                       float variance = MSE_DEFAULT_VARIANCE)
        {
            NGRAPH_CHECK(actual.size() == expected.size(),
                         "mse requires that actual and expected sizes match");
        }
    }
}
