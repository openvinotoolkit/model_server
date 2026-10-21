//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // NOLINT

namespace {
using Clock = std::chrono::steady_clock;
volatile unsigned char checksum = 0;

struct Decoded {
    std::unique_ptr<void, decltype(&stbi_image_free)> owner{nullptr, stbi_image_free};
    cv::Mat image;
};

Decoded decodeStb(const std::vector<unsigned char>& bytes) {
    int width = 0, height = 0, channels = 0;
    const int size = static_cast<int>(bytes.size());
    const bool is16 = stbi_is_16_bit_from_memory(bytes.data(), size);
    void* pixels = is16 ? static_cast<void*>(stbi_load_16_from_memory(bytes.data(), size, &width, &height, &channels, 0)) : static_cast<void*>(stbi_load_from_memory(bytes.data(), size, &width, &height, &channels, 0));
    std::unique_ptr<void, decltype(&stbi_image_free)> owner(pixels, stbi_image_free);
    if (!pixels)
        throw std::runtime_error(stbi_failure_reason());
    cv::Mat result(height, width, CV_MAKETYPE(is16 ? CV_16U : CV_8U, channels), pixels);
    if (channels >= 3) {
        const size_t channelBytes = result.elemSize1();
        for (size_t i = 0; i < result.total(); ++i) {
            for (size_t j = 0; j < channelBytes; ++j)
                std::swap(result.data[i * result.elemSize() + j], result.data[i * result.elemSize() + 2 * channelBytes + j]);
        }
    }
    return {std::move(owner), std::move(result)};
}

template <typename Decode>
void measure(Decode decode, std::vector<double>& samples) {
    for (int i = 0; i < 20; ++i) {
        const auto start = Clock::now();
        {
            Decoded decoded = decode();
            if (decoded.image.empty())
                throw std::runtime_error("Decoding failed");
            checksum = decoded.image.data[0];
        }
        samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
    }
}

double percentile(std::vector<double> samples, double fraction) {
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>((samples.size() - 1) * fraction)];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: image_decode_benchmark image [image ...]\n";
        return 1;
    }
    try {
        cv::setNumThreads(1);
        std::cerr << "OpenCV " << CV_VERSION << "; single worker; warmup=10; interleaved blocks=10x20; allocation/free included\n";
        std::cout << "file,width,height,channels,depth,backend,median_us,p95_us,images_per_second,max_abs_pixel_difference\n";
        for (int file = 1; file < argc; ++file) {
            std::ifstream input(argv[file], std::ios::binary);
            if (!input)
                throw std::runtime_error(std::string("Cannot read ") + argv[file]);
            const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            if (bytes.empty() || bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("Invalid encoded image size");
            auto opencv = [&bytes] {
                Decoded decoded;
                decoded.image = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
                return decoded;
            };
            auto stb = [&bytes] { return decodeStb(bytes); };
            auto original = [&bytes] {
                std::vector<unsigned char> copy(bytes.begin(), bytes.end());
                cv::Mat buffer(copy, true);
                Decoded decoded;
                decoded.image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
                return decoded;
            };
            Decoded referenceOwner = opencv();
            Decoded candidateOwner = stb();
            const cv::Mat& reference = referenceOwner.image;
            const cv::Mat& candidate = candidateOwner.image;
            if (reference.empty() || reference.size() != candidate.size() || reference.type() != candidate.type())
                throw std::runtime_error("Decoder output shape/type mismatch");
            const double difference = cv::norm(reference, candidate, cv::NORM_INF);
            for (int i = 0; i < 10; ++i) {
                opencv();
                stb();
                original();
            }
            std::vector<double> cvTimes, stbTimes, originalTimes;
            for (int block = 0; block < 10; ++block) {
                if (block % 2 == 0) {
                    measure(opencv, cvTimes);
                    measure(stb, stbTimes);
                } else {
                    measure(stb, stbTimes);
                    measure(opencv, cvTimes);
                }
                measure(original, originalTimes);
            }
            auto report = [&](const char* backend, const std::vector<double>& times) {
                double median = percentile(times, 0.5);
                std::cout << argv[file] << ',' << reference.cols << ',' << reference.rows << ','
                          << reference.channels() << ',' << reference.elemSize1() * 8 << ',' << backend << ','
                          << std::fixed << std::setprecision(2) << median << ',' << percentile(times, 0.95)
                          << ',' << 1e6 / median << ',' << difference << '\n';
            };
            report("opencv", cvTimes);
            report("stb_bgr", stbTimes);
            report("opencv_current_wrapper", originalTimes);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
