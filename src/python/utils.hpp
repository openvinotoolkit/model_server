#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

namespace ovms {

class PyObjectWrapper {
    std::unique_ptr<py::object> obj;
public:
    PyObjectWrapper() = delete;
    PyObjectWrapper(const PyObjectWrapper& other) = delete;
    PyObjectWrapper(const py::object& other);
    ~PyObjectWrapper();

    const py::object& getObject() const;

    template <typename T> 
    T getProperty(const std::string& name) const {
        py::gil_scoped_acquire acquire;
        try {
            T property = obj->attr(name.c_str()).cast<T>();
            return property;
        } catch (const pybind11::error_already_set& e) {
            std::cout << "PyObjectWrapper::getProperty failed: " << e.what() << std::endl;
            throw e;
        } catch (std::exception& e) {
            std::cout << "PyObjectWrapper::getProperty failed: " << e.what() << std::endl;
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
        this->message = "Unexpected Python object type. Expected: " + expectedType
                        + ". Received: " + objectType;
    };

    const char* what() const throw() override
    {
        return message.c_str();
    };
};


} // namespace ovms