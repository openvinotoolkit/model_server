//
// Created by Dariusz Trawinski on 2/17/21.
//
#include <iostream>
extern "C" {
#include "../../custom_node_interface.h"
}
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

auto reorder_to_chw(cv::Mat* mat) {
    assert(mat->channels() == 3);
    std::vector<float> data(mat->channels() * mat->rows * mat->cols);
    for(int y = 0; y < mat->rows; ++y) {
        for(int x = 0; x < mat->cols; ++x) {
            for(int c = 0; c < mat->channels(); ++c) {
                data[c * (mat->rows * mat->cols) + y * mat->cols + x] = mat->at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return data;
}

std::unique_ptr<struct CustomNodeTensor[]> Jpeg2CustomNodeTensor(std::string jpegPath){
    cv::Mat image;
    image = cv::imread( jpegPath, 1 );
    cv::Mat image_32;
    image.convertTo(image_32, CV_32F);
    auto image_nchw = reorder_to_chw(image &);
    std::string tensor_name("image");
    std::vector<uint64_t> dims{ 1, 3, image.size().height, image.size().width };
    inputTensors[i].name = static_cast<const char*>(tensor_name.c_str());
    inputTensors[i].data = static_cast<uint8_t*>(image_nchw.data());
    inputTensors[i].dataLength = static_cast<uint64_t>(sizeof(image_nchw[0]) * image_nchw.size(););
    inputTensors[i].dims = static_cast<uint64_t*>(blob->getTensorDesc().getDims().data());
    inputTensors[i].dimsLength = static_cast<uint64_t>(blob->getTensorDesc().getDims().size());
    inputTensors[i].precision = CustomNodeTensorPrecision::FP32;
    return std::move(inputTensors);
}

std::unique_ptr<struct CustomNodeParam[]> createCustomNodeParamArray(const std::unordered_map<std::string, std::string>& paramMap) {
    if (paramMap.size() == 0) {
        return nullptr;
    }
    auto Parameters = std::make_unique<struct CustomNodeParam[]>(paramMap.size());
    int i = 0;
    for (const auto& [key, value] : paramMap) {
        Parameters[i].key = key.c_str();
        Parameters[i].value = value.c_str();
        i++;
    }
    return std::move(Parameters);
}

int main() {
    std::cout << "test" << std::endl;
    int value = 10;
    // void* ptr;
    // ptr = &value;
    // int number = release(ptr);
    // std::cout << "number:" << number;
    std::unique_ptr<struct CustomNodeTensor[]> inputs;
    inputs = Jpeg2CustomNodeTensor("example_client/images/bee.jpeg");
    uint64_t inputTensorsLength = 1;
    struct CustomNodeTensor* outputTensors = nullptr;
    int outputTensorsLength = 1;
    const std::unordered_map<std::string, std::string>& paramMap = {{"width","224"},{"height","224"}};
    int parametersLength = 1;
    std::unique_ptr<struct CustomNodeParam[]> parameters = createCustomNodeParamArray(&paramMap);
    execute( inputTensors.get(), inputTensorsLength, &outputTensors, &outputTensorsLength, parameters.get(), parametersLength);

    return 0;
}