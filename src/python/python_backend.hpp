#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "utils.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace ovms {

class PythonBackend {
    std::unique_ptr<py::module_> pyovmsModule;
    std::unique_ptr<py::object> tensorClass;
public:
    PythonBackend();
    ~PythonBackend();
    static bool createPythonBackend(PythonBackend** pythonBackend);

    bool createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, 
                            py::ssize_t size, std::unique_ptr<PyObjectWrapper>& outTensor);

};
} // namespace ovms