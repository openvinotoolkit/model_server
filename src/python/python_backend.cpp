#include "python_backend.hpp"
#include <iostream>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

bool PythonBackend::createPythonBackend(PythonBackend** pythonBackend) {
    py::gil_scoped_acquire acquire;
    try {
    *pythonBackend = new PythonBackend();
    std::cout << pythonBackend;
    } catch (const pybind11::error_already_set& e) {
        std::cout << "PythonBackend initialization failed: " << e.what() << std::endl;
        return false;
    } catch (std::exception& e) {
        std::cout << "PythonBackend initialization failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

PythonBackend::PythonBackend() {
    py::gil_scoped_acquire acquire;
    py::print("Creating python backend");
    pyovmsModule = std::make_unique<py::module_>(py::module_::import("pyovms"));
    tensorClass = std::make_unique<py::object>(pyovmsModule->attr("Tensor"));
}

PythonBackend::~PythonBackend() {
    py::gil_scoped_acquire acquire;
    tensorClass.reset();
    pyovmsModule.reset();
}

bool PythonBackend::createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, 
                                             const std::string& datatype, py::ssize_t size, std::unique_ptr<PyObjectWrapper>& outTensor) {
    py::gil_scoped_acquire acquire;
    try {
        py::object ovmsPyTensor = tensorClass->attr("create_from_data")(name, ptr, shape, datatype, size);
        outTensor = std::make_unique<PyObjectWrapper>(ovmsPyTensor);
        return true;
    } catch (const pybind11::error_already_set& e) {
        std::cout << "PythonBackend::createOvmsPyTensor - Py Error: " << e.what();
        return false;
    } catch (std::exception& e) {
        std::cout << "PythonBackend::createOvmsPyTensor - Error: " << e.what();
        return false;
    }
}
