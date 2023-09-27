#include "ovms_py_tensor.hpp"

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <numeric>

namespace py = pybind11;
using namespace ovms;

OvmsPyTensor::OvmsPyTensor(void *ptr, std::vector<py::ssize_t> shape, std::string datatype, size_t size) :
    ptr(ptr),
    shape(shape),
    ndim(shape.size()),
    format(),
    itemsize(),
    datatype(datatype),
    size(size)
{
    if (datatypeToBufferFormat.count(datatype)) {
        // Known format
        format = datatypeToBufferFormat.at(datatype);
    }
    else {
        // Unknown format, assuming raw binary
        format = datatypeToBufferFormat.at("UINT8");
    }
    itemsize = bufferFormatToItemsize.at(format);
    strides.insert(strides.begin(), itemsize);
    for (int i = 1; i < ndim; i++){
        py::ssize_t stride = shape[ndim-i] * strides[0];
        strides.insert(strides.begin(), stride);
    }
}

OvmsPyTensor::OvmsPyTensor(py::buffer_info bufferInfo) : 
    ptr(bufferInfo.ptr),
    shape(bufferInfo.shape),
    ndim(bufferInfo.ndim),
    format(bufferInfo.format),
    itemsize(bufferInfo.itemsize),
    strides(bufferInfo.strides)
{
    size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<py::ssize_t>()) * itemsize;
    datatype = format;
    if (bufferFormatToDatatype.count(format)) {
        datatype = bufferFormatToDatatype.at(format);
    }
}
