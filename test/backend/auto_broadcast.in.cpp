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
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename optype, typename itype, typename otype>
void check_auto_bcast(
    const std::vector<std::vector<itype>>& inputs,
    const std::vector<otype> output,
    const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY))
{
    auto iet = element::from<itype>();
    auto oet = element::from<otype>();

    if (std::is_same<itype, char>::value)
    {
        iet = element::boolean;
    }
    if (std::is_same<otype, char>::value)
    {
        oet = element::boolean;
    }
    auto A = make_shared<op::Parameter>(iet, Shape{2, 3});
    auto B = make_shared<op::Parameter>(iet, Shape{3});
    auto f = make_shared<Function>(make_shared<optype>(A, B, autob), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(iet, Shape{2, 3});
    shared_ptr<runtime::Tensor> b = backend->create_tensor(iet, Shape{3});
    shared_ptr<runtime::Tensor> result = backend->create_tensor(oet, Shape{2, 3});

    copy_data(a, inputs[0]);
    copy_data(b, inputs[1]);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close(read_vector<otype>(result), output));
}

NGRAPH_TEST(${BACKEND_NAME}, auto_bcast_binary_elementwise)
{
    check_auto_bcast<op::Add, float, float>({{1, 2, 3, 4, 5, 6}, {5, 6, 7}}, {6, 8, 10, 9, 11, 13});
    check_auto_bcast<op::Subtract, float, float>({{1, 2, 3, 4, 5, 6}, {5, 6, 7}},
                                                 {-4.f, -4.f, -4.f, -1.f, -1.f, -1.f});
    check_auto_bcast<op::Multiply, float, float>({{1, 2, 3, 4, 5, 6}, {5, 6, 7}},
                                                 {5, 12, 21, 20, 30, 42});
    check_auto_bcast<op::Divide, float, float>({{4, 5, 6, 7, 8, 9}, {1, 2, 3}},
                                               {4, 2.5f, 2, 7, 4, 3});
    check_auto_bcast<op::Maximum, float, float>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                {1, 5, 8, 4, 5, 8});
    check_auto_bcast<op::Minimum, float, float>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                {1, 2, 3, 1, 5, 6});
    check_auto_bcast<op::Power, float, float>({{1, 2, 3, 4, 5, 6}, {1, 2, 3}},
                                              {1, 4, 27, 4, 25, 216});

    check_auto_bcast<op::And, char, char>({{1, 0, 1, 0, 0, 1}, {1, 0, 1}}, {1, 0, 1, 0, 0, 1});
    check_auto_bcast<op::Or, char, char>({{1, 0, 1, 0, 1, 1}, {1, 0, 0}}, {1, 0, 1, 1, 1, 1});

    check_auto_bcast<op::Equal, uint8_t, char>({{1, 0, 1, 0, 1, 1}, {1, 0, 0}}, {1, 1, 0, 0, 0, 0});
    check_auto_bcast<op::Greater, float, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 0, 0, 1, 0, 0});
    check_auto_bcast<op::GreaterEq, float, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                 {1, 0, 0, 1, 1, 0});
    check_auto_bcast<op::Less, uint8_t, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 1, 1, 0, 0, 1});
    check_auto_bcast<op::LessEq, uint8_t, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                {1, 1, 1, 0, 1, 1});
    check_auto_bcast<op::NotEqual, uint8_t, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                  {0, 1, 1, 1, 0, 1});
}

NGRAPH_TEST(${BACKEND_NAME}, auto_bcast_binary_elementwise_pdpd)
{
    const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
    check_auto_bcast<op::Add, float, float>(
        {{1, 2, 3, 4, 5, 6}, {5, 6, 7}}, {6, 8, 10, 9, 11, 13}, autob);
    check_auto_bcast<op::Subtract, float, float>(
        {{1, 2, 3, 4, 5, 6}, {5, 6, 7}}, {-4.f, -4.f, -4.f, -1.f, -1.f, -1.f}, autob);
    check_auto_bcast<op::Multiply, float, float>(
        {{1, 2, 3, 4, 5, 6}, {5, 6, 7}}, {5, 12, 21, 20, 30, 42}, autob);
    check_auto_bcast<op::Divide, float, float>(
        {{4, 5, 6, 7, 8, 9}, {1, 2, 3}}, {4, 2.5f, 2, 7, 4, 3}, autob);
    check_auto_bcast<op::Maximum, float, float>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {1, 5, 8, 4, 5, 8}, autob);
    check_auto_bcast<op::Minimum, float, float>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {1, 2, 3, 1, 5, 6}, autob);
    check_auto_bcast<op::Power, float, float>(
        {{1, 2, 3, 4, 5, 6}, {1, 2, 3}}, {1, 4, 27, 4, 25, 216}, autob);
    check_auto_bcast<op::And, char, char>(
        {{1, 0, 1, 0, 0, 1}, {1, 0, 1}}, {1, 0, 1, 0, 0, 1}, autob);
    check_auto_bcast<op::Or, char, char>(
        {{1, 0, 1, 0, 1, 1}, {1, 0, 0}}, {1, 0, 1, 1, 1, 1}, autob);

    check_auto_bcast<op::Equal, uint8_t, char>(
        {{1, 0, 1, 0, 1, 1}, {1, 0, 0}}, {1, 1, 0, 0, 0, 0}, autob);
    check_auto_bcast<op::Greater, float, char>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 0, 0, 1, 0, 0}, autob);
    check_auto_bcast<op::GreaterEq, float, char>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {1, 0, 0, 1, 1, 0}, autob);
    check_auto_bcast<op::Less, uint8_t, char>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 1, 1, 0, 0, 1}, autob);
    check_auto_bcast<op::LessEq, uint8_t, char>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {1, 1, 1, 0, 1, 1}, autob);
    check_auto_bcast<op::NotEqual, uint8_t, char>(
        {{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 1, 1, 1, 0, 1}, autob);
}

NGRAPH_TEST(${BACKEND_NAME}, auto_bcast_binary_elementwise_pdpd_dynamic)
{
    auto pshape_a = PartialShape::dynamic();
    auto pshape_b = PartialShape::dynamic();
    auto a = make_shared<op::Parameter>(element::f32, pshape_a);
    auto b = make_shared<op::Parameter>(element::f32, pshape_b);

    op::AutoBroadcastSpec autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, -1);
    auto f = make_shared<Function>(make_shared<op::Add>(a, b, autob), ParameterVector{a, b});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto t_a = backend->create_tensor(element::f32, Shape{2, 3});
    auto t_b = backend->create_tensor(element::f32, Shape{3});
    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6});
    copy_data(t_b, vector<float>{5, 6, 7});
    ex->call_with_validate({t_r}, {t_a, t_b});
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3}));

    auto results = read_vector<float>(t_r);
    vector<float> expected_values{6, 8, 10, 9, 11, 13};
    EXPECT_TRUE(test::all_close_f(results, expected_values));

    // a shape {2, 3, 4, 5}, b shape {3, 4} axis = 1
    autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
    f = make_shared<Function>(make_shared<op::Add>(a, b, autob), ParameterVector{a, b});
    ex = backend->compile(f);
    t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    t_a = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    t_b = backend->create_tensor(element::f32, Shape{3, 4});
    copy_data(t_a, vector<float>(2 * 3 * 4 * 5, 1));
    copy_data(t_b, vector<float>(3 * 4, 1));
    ex->call_with_validate({t_r}, {t_a, t_b});
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3, 4, 5}));

    // a shape {2, 3, 4, 5}, b shape {3, 1} axis = 1
    t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    t_a = backend->create_tensor(element::f32, Shape{2, 3, 4, 5});
    t_b = backend->create_tensor(element::f32, Shape{3, 1});
    copy_data(t_a, vector<float>(2 * 3 * 4 * 5, 1));
    copy_data(t_b, vector<float>(3, 1));
    ex->call_with_validate({t_r}, {t_a, t_b});
    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3, 4, 5}));
}
