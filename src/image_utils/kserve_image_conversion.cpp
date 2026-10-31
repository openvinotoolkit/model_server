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
#include "kserve_image_conversion.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/round.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "opencv_decoder_api.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"
#include "stb_image.h"  // NOLINT

namespace ovms {
namespace image_utils {
namespace {

struct DecoderPlugin {
    DecodeOpenCvV1 decode = nullptr;
    std::string error;
    DecoderPlugin() {
        std::filesystem::path directory;
#ifdef _WIN32
        HMODULE module = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(&openCvImageDecoderUnavailableReason), &module)) {
            constexpr DWORD pathCapacity = 32768;
            std::vector<char> path(pathCapacity);
            const DWORD size = GetModuleFileNameA(module, path.data(), pathCapacity);
            if (size > 0 && size < path.size())
                directory = std::filesystem::path(path.data()).parent_path();
        }
        constexpr const char* filename = "ovms_opencv_image_decoder.dll";
#else
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&openCvImageDecoderUnavailableReason), &info) && info.dli_fname)
            directory = std::filesystem::absolute(info.dli_fname).parent_path();
        constexpr const char* filename = "libovms_opencv_image_decoder.so";
#endif
        if (directory.empty()) {
            error = "OpenCV image decoder is unavailable: cannot determine the OVMS library directory";
            SPDLOG_WARN("{}", error);
            return;
        }
        const std::vector<std::filesystem::path> candidates{
            directory / "lib" / filename, directory.parent_path() / "lib" / filename,
            directory / filename, directory / "image_utils" / filename};
        error = "OpenCV image decoder is unavailable (OpenCV libraries or the decoder plugin are missing or incompatible)";
        for (const auto& candidate : candidates) {
#ifdef _WIN32
            HMODULE handle = LoadLibraryExA(candidate.string().c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!handle) {
                error += "; " + candidate.string() + ": Windows loader error " + std::to_string(GetLastError());
                continue;
            }
            decode = reinterpret_cast<DecodeOpenCvV1>(GetProcAddress(handle, "ovmsDecodeOpenCvV1"));
#else
            void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) {
                const char* detail = dlerror();
                error += "; " + candidate.string() + ": " + (detail ? detail : "unknown loader error");
                continue;
            }
            decode = reinterpret_cast<DecodeOpenCvV1>(dlsym(handle, "ovmsDecodeOpenCvV1"));
#endif
            if (decode) {
                error.clear();
                SPDLOG_INFO("Using OpenCV image decoder: {}", candidate.string());
                return;
            }
            error += "; " + candidate.string() + ": missing ovmsDecodeOpenCvV1 entry point";
        }
        SPDLOG_INFO("{}; using stb_image for supported image formats", error);
    }
};

DecoderPlugin& plugin() {
    static DecoderPlugin instance;
    return instance;
}

struct StbAllocator {
    std::shared_ptr<void> pixels;
    size_t size;
    void* allocate(size_t bytes, size_t) const {
        if (bytes != size)
            throw std::runtime_error("Unexpected stb decoded image allocation size");
        return pixels.get();
    }
    void deallocate(void*, size_t, size_t) noexcept {}
    bool is_equal(const StbAllocator& other) const { return pixels == other.pixels; }
};

using PreprocessKey = std::tuple<ov::element::Type_t, ov::element::Type_t, size_t, size_t, size_t>;
struct Preprocessor {
    ov::CompiledModel model;
};

std::shared_ptr<Preprocessor> getPreprocessor(const ov::Tensor& source,
    const ov::element::Type& precision, size_t height, size_t width) {
    static ov::Core core;
    static std::mutex mutex;
    static std::list<std::pair<PreprocessKey, std::shared_ptr<Preprocessor>>> cache;
    const size_t channels = source.get_shape()[3];
    const PreprocessKey key{source.get_element_type(), precision, channels, height, width};
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->first == key) {
            auto entry = it->second;
            cache.splice(cache.begin(), cache, it);
            return entry;
        }
    }
    auto input = std::make_shared<ov::op::v0::Parameter>(precision,
        ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic(), static_cast<int64_t>(channels)});
    auto result = std::make_shared<ov::op::v0::Result>(input);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    ov::preprocess::PrePostProcessor processor(model);
    processor.input().tensor().set_element_type(source.get_element_type()).set_layout("NHWC");
    processor.input().model().set_layout("NHWC");
    auto& steps = processor.input().preprocess();
    if (source.get_element_type().is_real() && precision.is_integral_number()) {
        steps.custom([](const ov::Output<ov::Node>& node) {
            return std::make_shared<ov::op::v5::Round>(node, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        });
    }
    // OpenCV saturates narrowing integer conversions instead of wrapping.
    if (precision == ov::element::i8 || precision == ov::element::i16 || precision == ov::element::u8 || precision == ov::element::u16) {
        const double minimum = precision == ov::element::i8 ? -128 : precision == ov::element::i16 ? -32768 : 0;
        const double maximum = precision == ov::element::i8 ? 127 : precision == ov::element::i16 ? 32767 : precision == ov::element::u8 ? 255 : 65535;
        steps.custom([minimum, maximum](const ov::Output<ov::Node>& node) {
            return std::make_shared<ov::op::v0::Clamp>(node, minimum, maximum);
        });
    }
    steps.convert_element_type(precision);
    if (precision.is_integral_number())
        steps.convert_element_type(ov::element::f32);
    steps.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, height, width);
    if (precision.is_integral_number()) {
        steps.custom([](const ov::Output<ov::Node>& node) {
            return std::make_shared<ov::op::v5::Round>(node, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        });
        steps.convert_element_type(precision);
    }
    auto entry = std::make_shared<Preprocessor>();
    entry->model = core.compile_model(processor.build(), "CPU", ov::hint::inference_precision(ov::element::f32),
        ov::inference_num_threads(1), ov::num_streams(1));
    cache.emplace_front(key, entry);
    if (cache.size() > 32)
        cache.pop_back();
    return entry;
}
}  // namespace

const std::string& openCvImageDecoderUnavailableReason() {
    return plugin().error;
}

Status decodeImage(std::string_view encoded, ov::Tensor& tensor, ImageDecoder decoder) {
    tensor = ov::Tensor{};
    if (encoded.empty() || encoded.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        SPDLOG_DEBUG("Encoded image is empty or exceeds the decoder size limit");
        return Status(StatusCode::IMAGE_PARSING_FAILED, "Encoded image is empty or exceeds the decoder size limit");
    }
    if (decoder != ImageDecoder::STB) {
        auto& backend = plugin();
        if (backend.decode) {
            std::array<char, 1024> error{};
            if (backend.decode(reinterpret_cast<const unsigned char*>(encoded.data()), encoded.size(), &tensor, error.data(), error.size()))
                return StatusCode::OK;
            SPDLOG_DEBUG("OpenCV image decoding failed: {}", error.data());
            return Status(StatusCode::IMAGE_PARSING_FAILED, error.data());
        }
        if (decoder == ImageDecoder::OPENCV)
            return Status(StatusCode::IMAGE_PARSING_FAILED, backend.error);
    }
    const auto* data = reinterpret_cast<const unsigned char*>(encoded.data());
    const int size = static_cast<int>(encoded.size());
    int width = 0, height = 0, channels = 0;
    stbi_info_from_memory(data, size, &width, &height, &channels);
    const int desiredChannels = channels == 2 ? 4 : 0;
    const bool is16 = stbi_is_16_bit_from_memory(data, size);
    const bool isHdr = stbi_is_hdr_from_memory(data, size);
    void* pixels = isHdr ? static_cast<void*>(stbi_loadf_from_memory(data, size, &width, &height, &channels, desiredChannels)) : is16 ? static_cast<void*>(stbi_load_16_from_memory(data, size, &width, &height, &channels, desiredChannels)) : static_cast<void*>(stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels));
    std::shared_ptr<void> owner(pixels, stbi_image_free);
    if (!pixels) {
        const char* reason = stbi_failure_reason();
        std::string detail = std::string("stb_image could not decode the image: ") + (reason ? reason : "unknown image format");
        if (decoder == ImageDecoder::AUTO)
            detail += ". " + plugin().error + ". Install OpenCV and the OVMS OpenCV decoder plugin to enable additional image formats";
        SPDLOG_DEBUG("{}", detail);
        return Status(StatusCode::IMAGE_PARSING_FAILED, detail);
    }
    if (desiredChannels)
        channels = desiredChannels;
    const size_t channelBytes = isHdr ? sizeof(float) : is16 ? 2 : 1;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    auto* output = static_cast<unsigned char*>(pixels);
    if (channels >= 3) {
        for (size_t i = 0; i < pixelCount; ++i) {
            for (size_t j = 0; j < channelBytes; ++j)
                std::swap(output[i * channels * channelBytes + j], output[(i * channels + 2) * channelBytes + j]);
        }
    }
    tensor = ov::Tensor(isHdr ? ov::element::f32 : is16 ? ov::element::u16 : ov::element::u8,
        ov::Shape{1, static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)},
        StbAllocator{std::move(owner), pixelCount * channels * channelBytes});
    return StatusCode::OK;
}

Status preprocessImage(const ov::Tensor& source, ov::Tensor& destination,
    const ov::element::Type& precision, size_t height, size_t width) {
    if (!source || source.get_shape().size() != 4 || source.get_shape()[0] != 1 ||
        source.get_size() == 0 || height == 0 || width == 0) {
        SPDLOG_DEBUG("Image preprocessing requires a non-empty single-image NHWC tensor and positive target dimensions");
        return Status(StatusCode::INVALID_SHAPE, "Image preprocessing requires a non-empty single-image NHWC tensor and positive target dimensions");
    }
    if (precision != ov::element::u8 && precision != ov::element::u16 &&
        precision != ov::element::i8 && precision != ov::element::i16 && precision != ov::element::i32 &&
        precision != ov::element::f16 && precision != ov::element::f32 && precision != ov::element::f64) {
        SPDLOG_DEBUG("Unsupported destination image precision: {}", precision.get_type_name());
        return Status(StatusCode::INVALID_PRECISION, "Unsupported destination image precision: " + precision.get_type_name());
    }
    if (source.get_element_type() == precision && source.get_shape()[1] == height && source.get_shape()[2] == width) {
        destination = source;
        return StatusCode::OK;
    }
    try {
        auto entry = getPreprocessor(source, precision, height, width);
        ov::InferRequest request = entry->model.create_infer_request();
        request.set_input_tensor(source);
        destination = ov::Tensor(precision, ov::Shape{1, height, width, source.get_shape()[3]});
        request.set_output_tensor(destination);
        request.infer();
        return StatusCode::OK;
    } catch (const ov::Exception& exception) {
        SPDLOG_ERROR("OpenVINO image preprocessing failed: {}", exception.what());
        return Status(StatusCode::INTERNAL_ERROR, std::string("OpenVINO image preprocessing failed: ") + exception.what());
    }
}

}  // namespace image_utils
}  // namespace ovms
