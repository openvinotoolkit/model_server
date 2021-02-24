//
// Created by Dariusz Trawinski on 2/17/21.
//
#include <iostream>
extern "C" {
#include "../../custom_node_interface.h"
}
#include <vector>
#include <unordered_map>
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

std::unique_ptr<struct CustomNodeTensor[]> Jpeg2CustomNodeTensor(std::string jpegPath, cv::Mat& image, std::vector<uint64_t>& shape){
    std::cout << "started Jpeg2CustomNodeTensor" << std::endl;
    image = cv::imread( jpegPath, 1 );
    std::cout << "created mat from jpeg " << image.rows << ":" << image.cols << std::endl;
    cv::Mat image_32;
    image.convertTo(image_32, CV_32F);
    std::cout << "mat converted to fp32 " << image_32.rows << ":" << image_32.cols << std::endl;
    auto image_nchw = reorder_to_chw(&image);
    std::cout << "mat reordered to nchw vector " << image_nchw.size() << std::endl;
    std::string tensor_name("image");
    shape = { 1, 3, image.rows, image.cols };
    std::cout << "created dims vector " << shape.size() << std::endl;
    auto inputTensors = std::make_unique<struct CustomNodeTensor[]>(1);
    std::cout << "initialized inputTensors " << std::endl;
    inputTensors[0].name = static_cast<const char*>(tensor_name.c_str());
    std::cout << "set tensor name" << std::endl;
    inputTensors[0].data = reinterpret_cast<uint8_t*>(image_nchw.data());
    std::cout << "set tensor data" << std::endl;
    inputTensors[0].dataLength = static_cast<uint64_t>(sizeof(image_nchw[0]) * image_nchw.size());
    std::cout << "set tensor dataLength" << std::endl;
    inputTensors[0].dims = static_cast<uint64_t*>(shape.data());
    std::cout << "set dims" <<  shape[0] << shape[1] << shape[2] << shape[3] << std::endl;
    inputTensors[0].dimsLength = static_cast<uint64_t>(4);
    inputTensors[0].precision = CustomNodeTensorPrecision::FP32;
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
    std::unique_ptr<struct CustomNodeTensor[]> inputs;
    //inputs = Jpeg2CustomNodeTensor("example_client/images/bee.jpeg");
    cv::Mat image;
    std::vector<uint64_t> shape;
    inputs = Jpeg2CustomNodeTensor("/workspace/east_utils/bee.jpeg", image, shape);
    std::cout << "jpeg converted to custom tensor" << std::endl;
    uint64_t inputTensorsLength = 1;
    struct CustomNodeTensor* outputTensors = nullptr;
    int * outputTensorsLength;
    const std::unordered_map<std::string, std::string>& paramMap = {{"width","224"},{"height","224"}};
    int parametersLength = 3;
    std::unique_ptr<struct CustomNodeParam[]> parameters = createCustomNodeParamArray(paramMap);
    std::cout << "About to start executing custom node" << std::endl;
    execute( inputs.get(), inputTensorsLength, &outputTensors, outputTensorsLength, parameters.get(), parametersLength);

    return 0;
}