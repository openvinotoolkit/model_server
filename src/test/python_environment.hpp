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
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#pragma warning(pop)

namespace py = pybind11;

class PythonEnvironment : public testing::Environment {
    mutable std::unique_ptr<py::gil_scoped_release> GILScopedRelease;

public:
    void SetUp() override;
    void TearDown() override;
    void releaseGILFromThisThread() const;
    void reacquireGILForThisThread() const;
};
