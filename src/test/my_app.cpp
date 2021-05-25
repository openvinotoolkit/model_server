#include <iostream>
#include <inference_engine.hpp>

using namespace InferenceEngine;

void print_shape(const std::string& str, const SizeVector& vec) {
    std::cout << str << ": ";
    for (auto n : vec)
        std::cout << n << ",";
    std::cout << std::endl;
}

int main()
{
    std::cout << "1" << std::endl;
    InferenceEngine::Core ie;
    std::cout << "2" << std::endl;

    InferenceEngine::CNNNetwork network = ie.ReadNetwork("/workspace/resnet50-binary/1/resnet50-binary-0001.xml");
    // "0" FP32 1,3,224,224 NCHW

    network.getInputsInfo().find("0")->second->setLayout(Layout::NHWC);
    // "0" FP32 1,3,224,224 NHWC

    std::cout << "3" << std::endl;

    // auto shapes = network.getInputShapes();
    // shapes["0"] = {1,3,300,300};
    // network.reshape(shapes);
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");

    print_shape("TensorDesc Network", executable_network.GetInputsInfo().find("0")->second->getTensorDesc().getDims());
    print_shape("BlockingDesc Network", executable_network.GetInputsInfo().find("0")->second->getTensorDesc().getBlockingDesc().getBlockDims());

    auto ireq = executable_network.CreateInferRequest();

    InferenceEngine::TensorDesc desc;
    desc.setPrecision(InferenceEngine::Precision::FP32);
    desc.setDims({1, 224, 224, 3});
    desc.setLayout(Layout::ANY);
    auto blob = InferenceEngine::make_shared_blob<float>(desc);
    blob->allocate();

    print_shape("TensorDesc Blob before adjustment", blob->getTensorDesc().getDims());
    print_shape("BlockingDesc Blob before adjustment", blob->getTensorDesc().getBlockingDesc().getBlockDims());

    // Before EAST
    blob->getTensorDesc().setLayout(Layout::NHWC);
    blob->getTensorDesc().reshape({1, 3, 224, 224});

    print_shape("TensorDesc Blob after adjustment", blob->getTensorDesc().getDims());
    print_shape("BlockingDesc Blob after adjustment", blob->getTensorDesc().getBlockingDesc().getBlockDims());

    ireq.SetBlob("0", blob);

    ireq.Infer();

    auto out = ireq.GetBlob("1463");


    // for (auto& input : network.getInputsInfo()) {
    //     input->
    // }

    std::cout << "SUCCESS" << std::endl;
    return 0;
}
