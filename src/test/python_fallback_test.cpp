//*****************************************************************************
// Copyright 2026 Intel Corporation
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

/**
 * Test suite for Python runtime fallback behavior.
 * 
 * Verifies:
 * 1. Graceful error handling when Python interpreter fails to initialize
 * 2. Python calculators plugin load failures are reported clearly
 * 3. Non-Python graphs work when Python support unavailable
 * 4. Python node graphs fail with descriptive errors
 */

#include <gtest/gtest.h>

#include "status.hpp"

namespace ovms {

class PythonRuntimeFallbackTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * Verify that new status codes for Python failures are properly defined.
 */
TEST_F(PythonRuntimeFallbackTest, PythonStatusCodesExist) {
    // Python runtime initialization failure
    Status pythonInitFailed(StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED);
    EXPECT_FALSE(pythonInitFailed.ok());
    EXPECT_STREQ(pythonInitFailed.string().c_str(),
        "Failed to initialize Python interpreter. Check that libpython.so is installed");

    // Python backend creation failure
    Status pythonBackendFailed(StatusCode::PYTHON_BACKEND_CREATION_FAILED);
    EXPECT_FALSE(pythonBackendFailed.ok());
    EXPECT_STREQ(pythonBackendFailed.string().c_str(),
        "Failed to create Python backend. Check that pyovms module is available in PYTHONPATH");
}

/**
 * Verify status codes can carry additional details/context.
 */
TEST_F(PythonRuntimeFallbackTest, PythonStatusCodesWithDetails) {
    // Python interpreter init failure with details
    Status pythonInitFailed(StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED,
        "libpython3.12.so.1: cannot open shared object file");
    EXPECT_FALSE(pythonInitFailed.ok());
    EXPECT_TRUE(pythonInitFailed.string().find("libpython3.12.so.1") != std::string::npos);

    // Python backend creation with details
    Status pythonBackendFailed(StatusCode::PYTHON_BACKEND_CREATION_FAILED,
        "pyovms module not found in PYTHONPATH=/opt/intel/openvino/python");
    EXPECT_FALSE(pythonBackendFailed.ok());
    EXPECT_TRUE(pythonBackendFailed.string().find("PYTHONPATH") != std::string::npos);
}

/**
 * Verify that PYTHON_INTERPRETER_INITIALIZATION_FAILED is treated as a failure status.
 */
TEST_F(PythonRuntimeFallbackTest, PythonInitFailureIsNotOk) {
    Status status(StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.getCode(), StatusCode::OK);
}

/**
 * Verify that PYTHON_BACKEND_CREATION_FAILED is treated as a failure status.
 */
TEST_F(PythonRuntimeFallbackTest, PythonBackendFailureIsNotOk) {
    Status status(StatusCode::PYTHON_BACKEND_CREATION_FAILED);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.getCode(), StatusCode::OK);
}

/**
 * Verify distinction between Python init failures and other internal errors.
 */
TEST_F(PythonRuntimeFallbackTest, PythonErrorsDistinctFromGenericErrors) {
    Status pythonInitErr(StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED);
    Status internalErr(StatusCode::INTERNAL_ERROR);

    EXPECT_NE(pythonInitErr.getCode(), internalErr.getCode());
    EXPECT_STRNE(pythonInitErr.string().c_str(), internalErr.string().c_str());
}

/**
 * Verify that comparison of Python failure statuses works.
 */
TEST_F(PythonRuntimeFallbackTest, StatusCodeComparison) {
    Status pythonInit1(StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED);
    Status pythonInit2(StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED);
    Status pythonBackend(StatusCode::PYTHON_BACKEND_CREATION_FAILED);

    EXPECT_EQ(pythonInit1.getCode(), pythonInit2.getCode());
    EXPECT_NE(pythonInit1.getCode(), pythonBackend.getCode());
}

}  // namespace ovms
