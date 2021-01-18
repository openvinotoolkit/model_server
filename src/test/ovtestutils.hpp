//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "mock_iinferrequest.hpp"

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

inline tensorflow::DataType fromInferenceEnginePrecision(Precision precision) {
    switch (precision) {
    case Precision::FP32:
        return tensorflow::DataType::DT_FLOAT;
    case Precision::FP16:
        return tensorflow::DataType::DT_HALF;
    // case Precision::Q78:   return tensorflow::DataType::
    case Precision::I16:
        return tensorflow::DataType::DT_INT16;
    case Precision::U8:
        return tensorflow::DataType::DT_UINT8;
    case Precision::I8:
        return tensorflow::DataType::DT_INT8;
    case Precision::U16:
        return tensorflow::DataType::DT_UINT16;
    case Precision::I32:
        return tensorflow::DataType::DT_INT32;
    case Precision::I64:
        return tensorflow::DataType::DT_INT64;
    // case Precision::BIN:   return tensorflow::DataType::
    case Precision::BOOL:
        return tensorflow::DataType::DT_BOOL;
    default:
        throw "Not all types mapped yet";
    }
}

class MockIInferRequestFailingInSetBlob : public MockIInferRequest {
    InferenceEngine::StatusCode SetBlob(const char*, const Blob::Ptr&, ResponseDesc*) noexcept override {
        return InferenceEngine::StatusCode::UNEXPECTED;
    }
};

class MockBlob : public InferenceEngine::Blob {
public:
    using Ptr = std::shared_ptr<MockBlob>;
    MOCK_METHOD(size_t, element_size, (), (const, noexcept));
    MockBlob(const InferenceEngine::TensorDesc& tensorDesc) :
        Blob(tensorDesc) {
        to = const_cast<char*>("12345678");
        _allocator = details::make_pre_allocator(to, 8);
    }
    MOCK_METHOD(void, allocate, (), (noexcept));
    MOCK_METHOD(bool, deallocate, (), (noexcept));
    InferenceEngine::LockedMemory<void> buffer() noexcept {
        return LockedMemory<void>(_allocator.get(), to, 0);
    }
    MOCK_METHOD(InferenceEngine::LockedMemory<const void>, cbuffer, (), (const, noexcept));
    MOCK_METHOD(const std::shared_ptr<InferenceEngine::IAllocator>&, getAllocator, (), (const, noexcept));
    MOCK_METHOD(void*, getHandle, (), (const, noexcept));

private:
    std::shared_ptr<IAllocator> _allocator;
    char* to;
};

class MockIInferRequestProperGetBlob : public MockIInferRequest {
public:
    using Ptr = std::shared_ptr<MockIInferRequest>;
    MockIInferRequestProperGetBlob(const InferenceEngine::TensorDesc& tensorDesc) :
        MockIInferRequest() {
        mockBlobPtr = std::make_shared<NiceMock<MockBlob>>(tensorDesc);
    }
    MOCK_METHOD(InferenceEngine::StatusCode, GetBlob_mocked, (const char*, Blob::Ptr&, ResponseDesc*), (noexcept));

    InferenceEngine::StatusCode GetBlob(const char* c, Blob::Ptr& ptr, ResponseDesc* d) noexcept {
        // this is just to register GetBlob call
        this->GetBlob_mocked(c, ptr, d);
        ptr = mockBlobPtr;
        return InferenceEngine::StatusCode::OK;
    }

private:
    std::shared_ptr<NiceMock<MockBlob>> mockBlobPtr;
};
