//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <string>

#include <gtest/gtest.h>

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)   \
    {                                         \
        auto* err = C_API_CALL;               \
        if (err != nullptr) {                 \
            uint32_t code = 0;                \
            const char* msg = nullptr;        \
            OVMS_StatusGetCode(err, &code);   \
            OVMS_StatusGetDetails(err, &msg); \
            std::string smsg(msg);            \
            OVMS_StatusDelete(err);           \
            EXPECT_EQ(0, code) << smsg;       \
            ASSERT_EQ(err, nullptr) << smsg;  \
        }                                     \
    }

#define ASSERT_CAPI_STATUS_NOT_NULL(C_API_CALL) \
    {                                           \
        auto* err = C_API_CALL;                 \
        if (err != nullptr) {                   \
            OVMS_StatusDelete(err);             \
        } else {                                \
            ASSERT_NE(err, nullptr);            \
        }                                       \
    }

#define ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(C_API_CALL, EXPECTED_STATUS_CODE)                              \
    {                                                                                                          \
        auto* err = C_API_CALL;                                                                                \
        if (err != nullptr) {                                                                                  \
            uint32_t code = 0;                                                                                 \
            const char* details = nullptr;                                                                     \
            ASSERT_EQ(OVMS_StatusGetCode(err, &code), nullptr);                                                \
            ASSERT_EQ(OVMS_StatusGetDetails(err, &details), nullptr);                                          \
            ASSERT_NE(details, nullptr);                                                                       \
            ASSERT_EQ(code, static_cast<uint32_t>(EXPECTED_STATUS_CODE))                                       \
                << std::string{"wrong code: "} + std::to_string(code) + std::string{"; details: "} << details; \
            OVMS_StatusDelete(err);                                                                            \
        } else {                                                                                               \
            ASSERT_NE(err, nullptr);                                                                           \
        }                                                                                                      \
    }
