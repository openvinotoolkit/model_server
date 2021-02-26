#include <iostream>

#include <inference_engine.hpp>

using namespace InferenceEngine;

int main() {
    Core engine;
    CNNNetwork network(engine.ReadNetwork("/workspace/east_fp32/1/model.xml"));
    //CNNNetwork network(engine.ReadNetwork("/workspace/east/darek/1/east_resnet_v1_50.xml"));
    ExecutableNetwork execNet = engine.LoadNetwork(network, "CPU", {});

    auto inferRequest = execNet.CreateInferRequest();

    TensorDesc desc;
    desc.setPrecision(Precision::FP32);
    desc.setDims({1, 3, 1024, 1920});

    auto blob = make_shared_blob<float>(desc);
    blob->allocate();
    const std::string blobName{"input_images"};
    inferRequest.SetBlob(blobName, blob);

    try {
        inferRequest.SetCompletionCallback([&inferRequest]() {
            std::cout << "callback executed" << std::endl;
        });
        std::cout << "starting async" << std::endl;
        inferRequest.StartAsync();
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        std::cout << "::StartAsync InferenceEngineException: " << e.what()  << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "::StartAsync std::exception: " << e.what()  << std::endl;
        return 1;
    } catch (...) {
        std::cout << "::StartAsync exception"  << std::endl;
        return 1;
    }

    try {
        auto status = inferRequest.Wait(IInferRequest::RESULT_READY);
        std::cout << "Wait status: " << status << std::endl;
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        std::cout << "::Wait InferenceEngineException: " << e.what()  << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "::Wait std::exception: " << e.what()  << std::endl;
        return 1;
    } catch (...) {
        std::cout << "::Wait exception"  << std::endl;
        return 1;
    }

    std::cout << "Completed\n" << std::endl;
    return 0;
}
