//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "opencv2/opencv.hpp"

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

struct Entry {
    tensorflow::string imagePath;
    tensorflow::int64 expectedLabel;
};

struct BinaryData {
    std::shared_ptr<char[]> imageData;
    std::streampos fileSize;
    tensorflow::int64 expectedLabel;
};

struct CvMatData {
    cv::Mat image;
    tensorflow::int64 expectedLabel;
    tensorflow::string layout;
};

template <typename T>
std::vector<T> reorderVectorToNchw(const T* nhwcVector, int rows, int cols, int channels) {
    std::vector<T> nchwVector(rows * cols * channels);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                nchwVector[c * (rows * cols) + y * cols + x] = reinterpret_cast<const T*>(nhwcVector)[y * channels * cols + x * channels + c];
            }
        }
    }
    return nchwVector;
}

const cv::Mat reorderMatToNchw(cv::Mat* mat) {
    uint64_t channels = mat->channels();
    uint64_t rows = mat->rows;
    uint64_t cols = mat->cols;
    auto nchwVector = reorderVectorToNchw<float>((float*)mat->data, rows, cols, channels);

    cv::Mat image(rows, cols, CV_32FC3);
    std::memcpy(image.data, nchwVector.data(), nchwVector.size() * sizeof(float));
    return image;
}

bool readImagesList(const tensorflow::string& path, std::vector<Entry>& entries) {
    entries.clear();
    std::ifstream infile(path);
    tensorflow::string image = "";
    tensorflow::int64 label = 0;

    if (!infile.is_open()) {
        std::cout << "Failed to open " << path << std::endl;
        return false;
    }

    while (infile >> image >> label) {
        entries.emplace_back(Entry{image, label});
    }

    return true;
}

bool readImagesBinary(const std::vector<Entry>& entriesIn, std::vector<BinaryData>& entriesOut) {
    entriesOut.clear();

    for (const auto& entry : entriesIn) {
        std::ifstream imageFile(entry.imagePath, std::ios::binary);
        if (!imageFile.is_open()) {
            std::cout << "Failed to open " << entry.imagePath << std::endl;
            return false;
        }

        std::filebuf* pbuf = imageFile.rdbuf();
        auto fileSize = pbuf->pubseekoff(0, std::ios::end, std::ios::in);

        auto image = std::unique_ptr<char[]>(new char[fileSize]());

        pbuf->pubseekpos(0, std::ios::in);
        pbuf->sgetn(image.get(), fileSize);
        imageFile.close();

        entriesOut.emplace_back(BinaryData{std::move(image), fileSize, entry.expectedLabel});
    }

    return true;
}

bool readImagesCvMat(const std::vector<Entry>& entriesIn, std::vector<CvMatData>& entriesOut, const tensorflow::string& layout, tensorflow::int64 width, tensorflow::int64 height) {
    entriesOut.clear();

    for (const auto& entryIn : entriesIn) {
        CvMatData entryOut;
        entryOut.layout = layout;
        entryOut.expectedLabel = entryIn.expectedLabel;
        try {
            entryOut.image = cv::imread(entryIn.imagePath);
            if (entryOut.image.data == nullptr) {
                return false;
            }
        } catch (cv::Exception& ex) {
            return false;
        }
        entryOut.image.convertTo(entryOut.image, CV_32F);
        cv::resize(entryOut.image, entryOut.image, cv::Size(width, height));
        if (layout == "nchw") {
            entryOut.image = reorderMatToNchw(&entryOut.image);
        }
        entriesOut.emplace_back(entryOut);
    }

    return true;
}
