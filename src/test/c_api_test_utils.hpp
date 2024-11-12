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

#include "../ovms.h"  // NOLINT
#include "test_utils.hpp"

#define THROW_ON_ERROR_CAPI(C_API_CALL)                                        \
    {                                                                          \
        auto* err = C_API_CALL;                                                \
        if (err != nullptr) {                                                  \
            uint32_t code = 0;                                                 \
            const char* msg = nullptr;                                         \
            OVMS_StatusCode(err, &code);                                       \
            OVMS_StatusDetails(err, &msg);                                     \
            std::string smsg(msg);                                             \
            OVMS_StatusDelete(err);                                            \
            EXPECT_EQ(0, code) << smsg;                                        \
            EXPECT_EQ(err, nullptr) << smsg;                                   \
            throw std::runtime_error("Error during C-API call: " #C_API_CALL); \
        }                                                                      \
    }

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)  \
    {                                        \
        auto* err = C_API_CALL;              \
        if (err != nullptr) {                \
            uint32_t code = 0;               \
            const char* msg = nullptr;       \
            OVMS_StatusCode(err, &code);     \
            OVMS_StatusDetails(err, &msg);   \
            std::string smsg(msg);           \
            OVMS_StatusDelete(err);          \
            EXPECT_EQ(0, code) << smsg;      \
            ASSERT_EQ(err, nullptr) << smsg; \
        }                                    \
    }

#define EXPECT_CAPI_STATUS_NULL(C_API_CALL)  \
    {                                        \
        auto* err = C_API_CALL;              \
        if (err != nullptr) {                \
            uint32_t code = 0;               \
            const char* msg = nullptr;       \
            OVMS_StatusCode(err, &code);     \
            OVMS_StatusDetails(err, &msg);   \
            std::string smsg(msg);           \
            OVMS_StatusDelete(err);          \
            EXPECT_EQ(0, code) << smsg;      \
            EXPECT_EQ(err, nullptr) << smsg; \
        }                                    \
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
            ASSERT_EQ(OVMS_StatusCode(err, &code), nullptr);                                                   \
            ASSERT_EQ(OVMS_StatusDetails(err, &details), nullptr);                                             \
            ASSERT_NE(details, nullptr);                                                                       \
            ASSERT_EQ(code, static_cast<uint32_t>(EXPECTED_STATUS_CODE))                                       \
                << std::string{"wrong code: "} + std::to_string(code) + std::string{"; details: "} << details; \
            OVMS_StatusDelete(err);                                                                            \
        } else {                                                                                               \
            ASSERT_NE(err, nullptr);                                                                           \
        }                                                                                                      \
    }

struct ServerSettingsGuard {
    ServerSettingsGuard(int port) {
        THROW_ON_ERROR_CAPI(OVMS_ServerSettingsNew(&settings));
        THROW_ON_ERROR_CAPI(OVMS_ServerSettingsSetGrpcPort(settings, port));
    }
    ~ServerSettingsGuard() {
        if (settings)
            OVMS_ServerSettingsDelete(settings);
    }
    OVMS_ServerSettings* settings{nullptr};
};
struct ModelsSettingsGuard {
    ModelsSettingsGuard(const std::string& configPath) {
        THROW_ON_ERROR_CAPI(OVMS_ModelsSettingsNew(&settings));
        THROW_ON_ERROR_CAPI(OVMS_ModelsSettingsSetConfigPath(settings, configPath.c_str()));
    }
    ~ModelsSettingsGuard() {
        if (settings)
            OVMS_ModelsSettingsDelete(settings);
    }
    OVMS_ModelsSettings* settings{nullptr};
};

struct ServerGuard {
    ServerGuard(const std::string configPath) {
        std::string port = "9000";
        randomizePort(port);
        ServerSettingsGuard serverSettingsGuard(std::stoi(port));
        ModelsSettingsGuard modelsSettingsGuard(configPath);
        THROW_ON_ERROR_CAPI(OVMS_ServerNew(&server));
        THROW_ON_ERROR_CAPI(OVMS_ServerStartFromConfigurationFile(server, serverSettingsGuard.settings, modelsSettingsGuard.settings));
    }
    ~ServerGuard() {
        if (server)
            OVMS_ServerDelete(server);
    }
    OVMS_Server* server{nullptr};
};
struct CallbackUnblockingStruct {
    std::promise<uint32_t> signal;
    void* bufferAddr = nullptr;
};
void callbackMarkingItWasUsedWith42(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct);
void checkDummyResponse(OVMS_InferenceResponse* response, double expectedValue, double tolerance);
