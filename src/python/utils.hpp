#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
        SPDLOG_DEBUG("PyObjectWrapper constructor start");
        obj = std::make_unique<T>(other);
        SPDLOG_DEBUG("PyObjectWrapper constructor end");
    };

    ~PyObjectWrapper() {
        py::gil_scoped_acquire acquire;
        SPDLOG_DEBUG("PyObjectWrapper destructor start ");
        obj.reset();
        SPDLOG_DEBUG("PyObjectWrapper destructor end ");
    }

    const T& getImmutableObject() const {
        py::gil_scoped_acquire acquire;
        if (obj) {
            return *obj;
        } else {
            throw std::exception();
        }
    };

    T& getMutableObject() const {
        py::gil_scoped_acquire acquire;
        if (obj) {
            return *obj;
        } else {
            throw std::exception();
        }
    };

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
    std::string message;

public:
    UnexpectedPythonObjectError() = delete;
    UnexpectedPythonObjectError(const py::object& obj, const std::string& expectedType) {
        py::gil_scoped_acquire acquire;
        std::string objectType = obj.attr("__class__").attr("__name__").cast<std::string>();
        this->message = "Unexpected Python object type. Expected: " + expectedType + ". Received: " + objectType;
    };

    const char* what() const throw() override {
        return message.c_str();
    };
};

}  // namespace ovms
