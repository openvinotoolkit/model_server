#include <inference_engine.hpp>

using namespace InferenceEngine; 

int main()
{
    auto core = std::make_unique<Core>();

    std::cout << "1" << std::endl;

    auto network = std::make_unique<CNNNetwork>(core->ReadNetwork("/ovms/src/test/dummy/1/dummy.xml"));

    std::cout << "2" << std::endl;

    auto executable_network = std::make_shared<ExecutableNetwork>(core->LoadNetwork(*network, "CPU", {}));

    std::cout << "3" << std::endl;

    TensorDesc desc;
    desc.setDims({1, 10});
    desc.setPrecision(Precision::FP32);
    desc.setLayout(Layout::ANY);

    std::cout << "4" << std::endl;

    auto blob = make_shared_blob<float>(desc);

    std::cout << "5" << std::endl;

    //blob->allocate();

    //((float*)blob->buffer())[0] = 7.3;

    std::cout << "6" << std::endl;

    auto infer_request = executable_network->CreateInferRequest();

    std::cout << "7" << std::endl;

    //infer_request.SetBlob("b", blob);

    std::cout << "8" << std::endl;

    infer_request.Infer();

    std::cout << "9" << std::endl;

    auto result = infer_request.GetBlob("a");

    //std::cout << ((float*)result->buffer())[0] << std::endl;
    std::cout << "Completed" << std::endl;
    return 0;
}
