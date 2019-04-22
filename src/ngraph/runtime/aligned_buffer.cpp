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
#include <memory>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

runtime::AlignedBuffer::AlignedBuffer()
    : m_allocated_buffer(nullptr)
    , m_aligned_buffer(nullptr)
    , m_byte_size(0)
{
}

runtime::AlignedBuffer::AlignedBuffer(size_t byte_size, size_t alignment)
{
    m_byte_size = std::max<size_t>(1, byte_size);
    size_t allocation_size = m_byte_size + alignment;
    m_allocated_buffer = static_cast<char*>(ngraph_malloc(allocation_size));
    m_aligned_buffer = m_allocated_buffer;
    size_t mod = size_t(m_aligned_buffer) % alignment;

    if (mod != 0)
    {
        m_aligned_buffer += (alignment - mod);
    }
}

runtime::AlignedBuffer::~AlignedBuffer()
{
    if (m_allocated_buffer != nullptr)
    {
        ngraph_free(m_allocated_buffer);
    }
}

std::shared_ptr<void> runtime::AlignedBuffer::make_aligned_ptr(size_t byte_size, size_t alignment)
{
    auto tmp = std::make_shared<AlignedBuffer>(byte_size, alignment);
    // Using the aliasing constructor for shared_ptr. The first argument holds the managed
    // object and the second, the pointer which is returned to the user of shared_ptr.
    return std::shared_ptr<void>(tmp, tmp->get_ptr());
}

std::shared_ptr<void> runtime::AlignedBuffer::make_aligned_ptr(void* buffer)
{
    std::shared_ptr<void> tmp;
    // Using the aliasing constructor for shared_ptr. The first argument holds the managed
    // object and the second, the pointer which is returned to the user of shared_ptr.
    return std::shared_ptr<void>(tmp, buffer);
}
