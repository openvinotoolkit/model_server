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

#include <memory>
#include <string>
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma warning(pop)

#include "../logging.hpp"
namespace py = pybind11;
using namespace py::literals;

namespace ovms {

template <class T>
class PyObjectWrapper {
    std::unique_ptr<T> obj;

public:
    PyObjectWrapper() = delete;
    PyObjectWrapper(const PyObjectWrapper& other) = delete;

    PyObjectWrapper(const T& other) {
        py::gil_scoped_acquire acquire;
        SPDLOG_TRACE("PyObjectWrapper constructor start");
        obj = std::make_unique<T>(other);
        SPDLOG_TRACE("PyObjectWrapper constructor end");
    }

    ~PyObjectWrapper() {
        py::gil_scoped_acquire acquire;
        SPDLOG_TRACE("PyObjectWrapper destructor start ");
        obj.reset();
        SPDLOG_TRACE("PyObjectWrapper destructor end ");
    }

    T& getObject() const {
        py::gil_scoped_acquire acquire;
        if (obj) {
            return *obj;
        } else {
            throw std::exception();
        }
    }

    template <typename U>
    inline U getProperty(const std::string& name) const {
        py::gil_scoped_acquire acquire;
        try {
            U property = obj->attr(name.c_str()).template cast<U>();
            return property;
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_DEBUG("PyObjectWrapper::getProperty failed: {}", e.what());
            throw e;
        } catch (std::exception& e) {
            SPDLOG_DEBUG("PyObjectWrapper::getProperty failed: {}", e.what());
            throw e;
        }
    }
};

class UnexpectedPythonObjectError : public std::exception {
protected:
    std::string message;

public:
    UnexpectedPythonObjectError() = delete;
    UnexpectedPythonObjectError(const UnexpectedPythonObjectError& exception) {
        this->message = std::string(exception.what());
    }
    UnexpectedPythonObjectError(const py::object& obj, const std::string& expectedType) {
        py::gil_scoped_acquire acquire;
        std::string objectType = obj.attr("__class__").attr("__name__").cast<std::string>();
        this->message = "Unexpected Python object type. Expected: " + expectedType + ". Received: " + objectType;
    }

    const char* what() const throw() override {
        return message.c_str();
    }
};

class UnexpectedInputPythonObjectError : UnexpectedPythonObjectError {
public:
    UnexpectedInputPythonObjectError(const UnexpectedPythonObjectError& exception) :
        UnexpectedPythonObjectError(exception) {}
    const char* what() const throw() override { return UnexpectedPythonObjectError::what(); }
};
class UnexpectedOutputPythonObjectError : UnexpectedPythonObjectError {
public:
    UnexpectedOutputPythonObjectError(const UnexpectedPythonObjectError& exception) :
        UnexpectedPythonObjectError(exception) {}
    const char* what() const throw() override { return UnexpectedPythonObjectError::what(); }
};

class BadPythonNodeConfigurationError : public std::exception {
    std::string message;

public:
    BadPythonNodeConfigurationError() = delete;
    BadPythonNodeConfigurationError(const std::string& message) {
        this->message = "Bad python node configuration. " + message;
    }

    const char* what() const throw() override {
        return message.c_str();
    }
};

class UnexpectedOutputTensorError : public std::exception {
    std::string message;

public:
    UnexpectedOutputTensorError() = delete;
    UnexpectedOutputTensorError(const std::string& outputName) {
        this->message = "Unexpected Tensor found in the outputs. Tensor name: " + outputName + " is not a valid node output";
    }

    const char* what() const throw() override {
        return message.c_str();
    }
};

}  // namespace ovms
