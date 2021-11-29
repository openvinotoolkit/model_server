//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include <gmock/gmock-generated-function-mockers.h>

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using InferenceEngine::IInferRequest;
using InferenceEngine::Precision;
using InferenceEngine::PreProcessInfo;
using InferenceEngine::ResponseDesc;

using namespace ovms;
using namespace InferenceEngine;

using testing::_;
using testing::NiceMock;
using testing::Throw;

inline tensorflow::DataType fromInferenceEnginePrecision(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return tensorflow::DataType::DT_FLOAT;
    case InferenceEngine::Precision::FP16:
        return tensorflow::DataType::DT_HALF;
    // case InferenceEngine::Precision::Q78:   return tensorflow::DataType::
    case InferenceEngine::Precision::I16:
        return tensorflow::DataType::DT_INT16;
    case InferenceEngine::Precision::U8:
        return tensorflow::DataType::DT_UINT8;
    case InferenceEngine::Precision::I8:
        return tensorflow::DataType::DT_INT8;
    case InferenceEngine::Precision::U16:
        return tensorflow::DataType::DT_UINT16;
    case InferenceEngine::Precision::I32:
        return tensorflow::DataType::DT_INT32;
    case InferenceEngine::Precision::I64:
        return tensorflow::DataType::DT_INT64;
    // case InferenceEngine::Precision::BIN:   return tensorflow::DataType::
    case InferenceEngine::Precision::BOOL:
        return tensorflow::DataType::DT_BOOL;
    default:
        throw "Not all types mapped yet";
    }
}

class MockBlob : public InferenceEngine::MemoryBlob {
public:
    using Ptr = std::shared_ptr<MockBlob>;
    MOCK_METHOD(size_t, byteSize, (), (const, noexcept));
    MockBlob(const InferenceEngine::TensorDesc& tensorDesc) :
        MemoryBlob(tensorDesc) {
        to = const_cast<char*>("12345678");
        _allocator = details::make_pre_allocator(to, 8);
    }
    MOCK_METHOD(void, allocate, (), (noexcept));
    MOCK_METHOD(bool, deallocate, (), (noexcept));
    InferenceEngine::LockedMemory<void> buffer() noexcept {
        return LockedMemory<void>(_allocator.get(), to, 0);
    }
    InferenceEngine::LockedMemory<void> rwmap() noexcept {
        return LockedMemory<void>(_allocator.get(), to, 0);
    }
    InferenceEngine::LockedMemory<const void> rmap() const noexcept {
        return LockedMemory<const void>(_allocator.get(), to, 0);
    }
    MOCK_METHOD(InferenceEngine::LockedMemory<const void>, cbuffer, (), (const, noexcept));
    MOCK_METHOD(InferenceEngine::LockedMemory<void>, wmap, (), (noexcept));
    MOCK_METHOD(const std::shared_ptr<InferenceEngine::IAllocator>&, getAllocator, (), (const, noexcept));
    MOCK_METHOD(void*, getHandle, (), (const, noexcept));

private:
    std::shared_ptr<IAllocator> _allocator;
    char* to;
};

class MockBlob_2 : public ov::runtime::Tensor {
public:
    MockBlob_2(const std::shared_ptr<ovms::TensorInfo>& info) :
        ov::runtime::Tensor(info->getOvPrecision(), info->getShape_2()) {
        to = const_cast<char*>("12345678");
    }

    // TODO: Those are not virtual methods, therefore mocks do not work.
    MOCK_METHOD(ov::Shape, get_shape, (), (const));
    MOCK_METHOD(size_t, get_byte_size, (), (const));
    MOCK_METHOD(ov::element::Type, get_element_type, (), (const));

private:
    char* to;
};
