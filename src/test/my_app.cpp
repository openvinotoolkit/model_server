#include <iostream>
#include <inference_engine.hpp>

using namespace InferenceEngine;

void print_shape(const std::string& str, const SizeVector& vec) {
    std::cout << str << ": ";
    for (auto n : vec)
        std::cout << n << ",";
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "1" << std::endl;
    InferenceEngine::Core ie;
    std::cout << "2" << std::endl;

    std::string model_path = argv[1];

    //InferenceEngine::CNNNetwork network = ie.ReadNetwork("/workspace/resnet50-tf-int8/1/resnet50-tf-int8.xml");
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(model_path);
    // "0" FP32 1,3,224,224 NCHW

    std::string inputName = network.getInputsInfo().begin()->first;
    std::string outputName = network.getOutputsInfo().begin()->first;

    network.getInputsInfo().find(inputName)->second->setLayout(Layout::NHWC);
    // "0" FP32 1,3,224,224 NHWC

    std::cout << "3" << std::endl;

    // auto shapes = network.getInputShapes();
    // shapes["0"] = {1,3,300,300};
    // network.reshape(shapes);
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");

    std::cout << "4" << std::endl;

    print_shape("TensorDesc Network", executable_network.GetInputsInfo().find(inputName)->second->getTensorDesc().getDims());
    print_shape("BlockingDesc Network", executable_network.GetInputsInfo().find(inputName)->second->getTensorDesc().getBlockingDesc().getBlockDims());

    auto ireq = executable_network.CreateInferRequest();
    std::cout << "5" << std::endl;

    InferenceEngine::TensorDesc desc;
    desc.setPrecision(InferenceEngine::Precision::FP32);
    desc.setDims({1, 224, 224, 3});
    desc.setLayout(Layout::ANY);
    auto blob = InferenceEngine::make_shared_blob<float>(desc);
    blob->allocate();
    std::cout << "6" << std::endl;

    print_shape("TensorDesc Blob before adjustment", blob->getTensorDesc().getDims());
    print_shape("BlockingDesc Blob before adjustment", blob->getTensorDesc().getBlockingDesc().getBlockDims());

    // Before EAST
    blob->getTensorDesc().setLayout(Layout::NHWC);
    blob->getTensorDesc().reshape({1, 3, 224, 224});

    print_shape("TensorDesc Blob after adjustment", blob->getTensorDesc().getDims());
    print_shape("BlockingDesc Blob after adjustment", blob->getTensorDesc().getBlockingDesc().getBlockDims());
    std::cout << "7" << std::endl;

    ireq.SetBlob(inputName, blob);
    std::cout << "8" << std::endl;

    ireq.Infer();
    std::cout << "9" << std::endl;

    auto out = ireq.GetBlob(outputName);

    std::cout << "10" << std::endl;

    // for (auto& input : network.getInputsInfo()) {
    //     input->
    // }

    std::cout << ((char*)out->buffer())[0] << std::endl;

    std::cout << "SUCCESS" << std::endl;
    return 0;
}
