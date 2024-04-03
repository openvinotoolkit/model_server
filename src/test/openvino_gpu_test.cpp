#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"

using namespace ov;
using std::cout, std::endl;

#define LOG(A) std::cout << __LINE__ << ":" << (A) << std::endl;

int main(int argc, char** argv) {
    cl_int err;

    // Step 1: Querying Platforms
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms\n";
        return 1;
    }

    cl_platform_id platform_id;
    clGetPlatformIDs(1, &platform_id, nullptr);

    // Step 2: Querying Devices
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of devices\n";
        return 1;
    }

    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting GPU device_id\n";
        return 1;
    }

    // Step 3: Creating a Context
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context\n";
        return 1;
    }

    Core core;
    auto model = core.read_model("dummy/1/dummy.xml");
    // get opencl queue object
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, context, 0);
    auto compiledModel = core.compile_model(model, remote_context);
    auto inferRequest = compiledModel.create_infer_request();
    // get context from plugin
    auto ov_context = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
    {
        auto context_from_plugin = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    }
    // Extract ocl context handle from RemoteContext
    cl_context context_handle = ov_context.get();
    auto devices = ov_context.get_device_name();
    std::cout << devices << std::endl;
    auto params = ov_context.get_params();
    for (auto [k, v] : params) {
        std::cout << k << ":" << v.as<std::string>() << std::endl;
    }
    std::cout << "OVMS_Status* OVMS_InferenceRequestInputSetData(OVMS_InferenceRequest* request, const char* inputName, const void* data, size_t byteSize, OVMS_BufferType bufferType, uint33_t deviceId);" << std::endl;
    // create the OpenCL buffers within the context

    // share the queue with GPU plugin and cnfer request$
    //  98     auto shared_in_blob = remote_context.create_tensor(input->get_element_type(), input->get_shape(), shared_in_buffer);$
    //   99     auto shared_out_blob = remote_context.create_tensor(output->get_element_type(), output->get_shape(), shared_out_buffer);ompile model
    auto exec_net_shared = core.compile_model(model, remote_context);

    auto input = model->get_parameters().at(0);
    auto input_size = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto output_size = ov::shape_size(output->get_shape());
    // we need byte size not no of elements
    input_size *= sizeof(float);
    output_size *= sizeof(float);

    LOG(input_size);
    LOG(output_size);
    // create the OpenCL buffers within the context
    cl::Context cpp_cl_context(context);  // Convert cl_context to cl::Context
    cl::Buffer shared_in_buffer(cpp_cl_context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    cl::Buffer shared_out_buffer(cpp_cl_context, CL_MEM_READ_WRITE, output_size, NULL, &err);
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto shared_in_blob = remote_context.create_tensor(input->get_element_type(), input->get_shape(), shared_in_buffer);
    auto shared_out_blob = remote_context.create_tensor(output->get_element_type(), output->get_shape(), shared_out_buffer);
    std::vector<float> in(10, 0.1);
    void* buffer_in = in.data();
    LOG("A");
    // we want to read buffer so we need to eqnueere Read
    cl_command_queue_properties queue_properties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    cl::Platform platform = cl::Platform(platform_id);
    cl::Device device(device_id);
    auto queue = cl::CommandQueue(cpp_cl_context, device, queue_properties);
    queue.enqueueWriteBuffer(shared_in_buffer, /*blocking*/ true, 0, input_size, buffer_in);
    inferRequest.set_tensor(input, shared_in_blob);
    inferRequest.set_tensor(output, shared_out_blob);
    // infer
    inferRequest.infer();
    LOG("A");
    std::vector<float> out(10);
    void* buffer_out = out.data();
    queue.enqueueReadBuffer(shared_out_buffer, /*blocking*/ true, 0, output_size, buffer_out);
    LOG("A");
    LOG("A");
    float* val = (float*)buffer_in;
    LOG("A");
    std::cout << "in tensor:";
    for (int i = 0; i < 10; ++i) {
        std::cout << *(val++) << ", ";
    }
    cout << endl;
    try {
        LOG("A");
        LOG("A");
        val = (float*)buffer_out;
        LOG("A");
        std::cout << "out tensor:";
        for (int i = 0; i < 10; ++i) {
            std::cout << *(val++) << ", ";
        }
        cout << endl;
    } catch (std::exception& e) {
        cout << "BLAD:" << e.what() << endl;
    }
}
