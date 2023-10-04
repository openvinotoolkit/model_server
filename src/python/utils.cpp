#include "python_backend.hpp"
#include <iostream>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;


PyObjectWrapper::PyObjectWrapper(const py::object& other) {
    py::gil_scoped_acquire acquire;
    std::cout << "PyObjectWrapper constructor start" << std::endl;
    obj = std::make_unique<py::object>(other);
    std::cout << "PyObjectWrapper constructor end" << std::endl;
};

PyObjectWrapper::~PyObjectWrapper() {
    py::gil_scoped_acquire acquire;
    std::cout << "PyObjectWrapper destructor start " << std::endl;
    obj.reset();
    std::cout << "PyObjectWrapper destructor end " << std::endl;
}

const py::object& PyObjectWrapper::getObject() const {
    py::gil_scoped_acquire acquire;
    if (obj) {
        return *obj;
    } else {
        throw std::exception();
    }
};
