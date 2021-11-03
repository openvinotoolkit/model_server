#include <inference_engine.hpp>
#include <iostream>


int main() {
    InferenceEngine::SizeVector shape{1, 4};
    std::cout << "1111" << std::endl;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("src/test/dummy/1/dummy.xml");
    if (network.getInputShapes()["b"] != shape) {
        ngraph::PartialShape p_shape({1, 10});
        p_shape[1] = ngraph::Dimension(1, 500);
        network.reshape({{"b", p_shape}});
    }
    std::cout << "2222" << std::endl;

    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
    std::cout << "22222.11111" << std::endl;
    auto ireq = executable_network.CreateInferRequest();
    std::cout << "33333" << std::endl;

    InferenceEngine::TensorDesc desc;
    desc.setPrecision(InferenceEngine::Precision::FP32);
    desc.setDims(shape);
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(desc);
    std::cout << "4444" << std::endl;
    
    blob->allocate();
    float* inputData = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap().as<float*>();
    std::cout << "input:  ";
    for (int i = 0; i < shape[1]; i++) {
        inputData[i] = i + 5.0;
        std::cout << inputData[i] << ",";
    }
    std::cout << "5555" << std::endl;

    std::cout << std::endl;
    ireq.SetBlob("b", blob);
    ireq.Infer();
    auto res =  ireq.GetBlob("a");
    std::cout << "66666" << std::endl;
    
    float* outputData = InferenceEngine::as<InferenceEngine::MemoryBlob>(res)->rmap().as<float*>();
    std::cout << "output: ";
    for (int i = 0; i < shape[1]; i++) {
        std::cout << outputData[i] << ",";
    }
    std::cout << std::endl;
    std::cout << "777777" << std::endl;

    return 0;
}
