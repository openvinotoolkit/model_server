#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../ovms_py_tensor.hpp"

#include <string>
#include <vector>

namespace py = pybind11;
using namespace ovms;

PYBIND11_MODULE(pyovms, m) {
    py::class_<OvmsPyTensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer([](OvmsPyTensor &m) -> py::buffer_info {
            return py::buffer_info(
                m.ptr,           
                m.itemsize,   
                m.format,
                m.ndim,
                m.shape,
                m.strides
            );
        })
        .def(py::init([](py::buffer b) {
            /* Request a buffer descriptor from Python */
            return OvmsPyTensor(b.request());
        }))
        .def_property_readonly("data",  [](const py::object &m) -> py::memoryview { return py::memoryview(m); })
        .def_property_readonly("shape", [](const OvmsPyTensor &m) -> py::tuple { return py::tuple(py::cast(m.shape)); } )
        .def_readonly("itemsize", &OvmsPyTensor::itemsize)
        .def_property_readonly("strides", [](const OvmsPyTensor &m) -> py::tuple { return py::tuple(py::cast(m.strides)); } )
        .def_readonly("ndim", &OvmsPyTensor::ndim)
        .def_readonly("format", &OvmsPyTensor::format)
        .def_readonly("size", &OvmsPyTensor::size)
        .def_readonly("datatype", &OvmsPyTensor::datatype);
}