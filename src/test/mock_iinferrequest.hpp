//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gmock/gmock-generated-function-mockers.h>

using InferenceEngine::IInferRequest;
using InferenceEngine::Precision;
using InferenceEngine::PreProcessInfo;
using InferenceEngine::ResponseDesc;

using namespace InferenceEngine;

class MockIInferRequest : public IInferRequest {
public:
    using Ptr = std::shared_ptr<MockIInferRequest>;
    MOCK_METHOD(InferenceEngine::StatusCode, StartAsync, (InferenceEngine::ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, SetBlob, (const char*, const Blob::Ptr&, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, SetBlob, (const char*, const Blob::Ptr&, const PreProcessInfo&, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(void, Release, (), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, Infer, (ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, Wait, (int64_t millis_timeout, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, GetUserData, (void**, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, SetUserData, (void*, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, SetCompletionCallback, (IInferRequest::CompletionCallback), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, GetBlob, (const char*, Blob::Ptr&, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, GetPreProcess, (const char*, const PreProcessInfo**, ResponseDesc*), (noexcept, const));
    MOCK_METHOD(InferenceEngine::StatusCode, SetBatch, (int batch, ResponseDesc*), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, GetPerformanceCounts, ((std::map<std::string, InferenceEngineProfileInfo> & perfMap), ResponseDesc*), (noexcept, const));
    MOCK_METHOD(InferenceEngine::StatusCode, QueryState, (IVariableState::Ptr & pState, size_t idx, ResponseDesc* resp), (noexcept, override));
    MOCK_METHOD(InferenceEngine::StatusCode, Cancel, (ResponseDesc*), (noexcept, override));
};
