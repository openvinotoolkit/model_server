//*****************************************************************************
// Copyright 2025 Intel Corporation
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

/**
 * @file imp_audio_utils.cpp
 * @brief Implementation of Intel MPI audio helpers for OVMS.
 *
 * All media operations go through the Intel MPI C API (intel_mpi.h).
 * This file does NOT call GStreamer directly — the MPI library owns
 * that abstraction.
 *
 * Two compile-time paths:
 *   1. INTEL_MPI_AVAILABLE — calls imp_decode_audio() etc.
 *   2. Fallback           — all functions return false / no-op
 */

#include "imp_audio_utils.hpp"

#ifdef INTEL_MPI_AVAILABLE

#include "intel_mpi.h"          // MPI C API

#include <openvino/openvino.hpp>  // ov::Tensor for data extraction
#include <cstring>
#include <iostream>

namespace ovms {
namespace imp {

bool isAvailable() {
    // imp_get_version succeeds when the library is linked in.
    // A more thorough check could try creating a dummy context,
    // but version presence is enough to confirm linkage.
    int major = 0, minor = 0, patch = 0;
    imp_get_version(&major, &minor, &patch);
    return (major > 0 || minor > 0 || patch > 0);
}

bool decodeAudioBuffer(const std::string_view& audioData,
                       uint32_t sampleRate,
                       std::vector<float>& pcm) {
    if (audioData.empty()) return false;

    // Configure decode options
    imp_audio_decode_opts_t opts{};
    opts.sample_rate = sampleRate;
    opts.channels    = 1;          // mono output for speech models
    opts.output_type = IMP_TYPE_FP32;
    opts.normalize   = true;

    // Decode via Intel MPI (context = NULL → no GPU, pure media decode)
    imp_tensor_t* tensor = nullptr;
    imp_status_t status = imp_decode_audio(
        &tensor,
        audioData.data(),
        audioData.size(),
        nullptr,           // ctx — not needed for CPU audio decode
        &opts,
        nullptr,           // callback — synchronous
        nullptr);          // user_data

    if (status != IMP_OK || !tensor) {
        if (tensor) imp_tensor_release(tensor);
        return false;
    }

    // Extract shape → number of float samples
    int64_t dims[4] = {};
    size_t ndims = 4;
    imp_tensor_get_shape(tensor, dims, &ndims);

    size_t numSamples = 1;
    for (size_t i = 0; i < ndims; i++) {
        numSamples *= static_cast<size_t>(dims[i]);
    }

    // Get pointer to underlying ov::Tensor data
    void* ov_tensor_ptr = nullptr;
    imp_device_type_t dev = IMP_DEVICE_CPU;
    imp_tensor_get_ov(tensor, &ov_tensor_ptr, &dev);

    if (!ov_tensor_ptr || dev != IMP_DEVICE_CPU) {
        imp_tensor_release(tensor);
        return false;
    }

    // The ov_tensor_ptr is an ov::Tensor* — extract float data
    auto* ovt = reinterpret_cast<ov::Tensor*>(ov_tensor_ptr);
    const float* fdata = ovt->data<float>();

    pcm.assign(fdata, fdata + numSamples);

    // Release the tensor through the MPI C API.
    imp_tensor_release(tensor);

    return !pcm.empty();
}

bool encodeToWav(const std::vector<float>& pcm,
                 uint32_t sampleRate,
                 uint16_t bitsPerSample,
                 std::vector<uint8_t>& wavData) {
    // WAV encoding is trivial — no need to route through GStreamer.
    // Keep this self-contained for minimal overhead.
    if (pcm.empty()) return false;
    (void)bitsPerSample;  // always output 32-bit float WAV

    uint32_t numChannels = 1;
    uint32_t byteRate = sampleRate * numChannels * sizeof(float);
    uint32_t blockAlign = numChannels * sizeof(float);
    uint32_t dataSize = static_cast<uint32_t>(pcm.size() * sizeof(float));
    uint32_t chunkSize = 36 + dataSize;

    wavData.clear();
    wavData.reserve(44 + dataSize);

    auto write16 = [&](uint16_t v) { wavData.push_back(v & 0xFF); wavData.push_back((v >> 8) & 0xFF); };
    auto write32 = [&](uint32_t v) {
        wavData.push_back(v & 0xFF); wavData.push_back((v >> 8) & 0xFF);
        wavData.push_back((v >> 16) & 0xFF); wavData.push_back((v >> 24) & 0xFF);
    };
    auto writeStr = [&](const char* s, size_t n) { wavData.insert(wavData.end(), s, s + n); };

    writeStr("RIFF", 4);
    write32(chunkSize);
    writeStr("WAVE", 4);
    writeStr("fmt ", 4);
    write32(16);              // subchunk1 size
    write16(3);               // IEEE float format
    write16(static_cast<uint16_t>(numChannels));
    write32(sampleRate);
    write32(byteRate);
    write16(static_cast<uint16_t>(blockAlign));
    write16(32);              // bits per sample
    writeStr("data", 4);
    write32(dataSize);

    const uint8_t* raw = reinterpret_cast<const uint8_t*>(pcm.data());
    wavData.insert(wavData.end(), raw, raw + dataSize);

    return true;
}

bool encodeAudioBuffer(const float* samples,
                       size_t numSamples,
                       uint32_t sampleRate,
                       uint16_t bitsPerSample,
                       const std::string& format,
                       std::vector<uint8_t>& output) {
    if (!samples || numSamples == 0) return false;

    // For WAV we can use the local writer (fast, no GStreamer init)
    if (format == "wav") {
        std::vector<float> pcm(samples, samples + numSamples);
        return encodeToWav(pcm, sampleRate, bitsPerSample, output);
    }

    // Route everything else through the MPI C API
    imp_audio_encode_opts_t opts{};
    opts.codec       = format.c_str();
    opts.sample_rate = sampleRate;
    opts.channels    = 1;
    opts.bitrate_kbps = 192;  // reasonable default for speech
    opts.output_path  = nullptr;

    void*  data = nullptr;
    size_t dataSize = 0;
    imp_status_t st = imp_encode_audio(&data, &dataSize, samples, numSamples, &opts);
    if (st != IMP_OK || !data) {
        if (data) imp_free(data);
        return false;
    }

    output.assign(static_cast<uint8_t*>(data),
                  static_cast<uint8_t*>(data) + dataSize);
    imp_free(data);
    return true;
}

}  // namespace imp
}  // namespace ovms

#else  // !INTEL_MPI_AVAILABLE

// ---- Stub implementation (Linux / no GStreamer) ------------------------------

namespace ovms {
namespace imp {

bool isAvailable() { return false; }

bool decodeAudioBuffer(const std::string_view& /*audioData*/,
                       uint32_t /*sampleRate*/,
                       std::vector<float>& /*pcm*/) {
    return false;
}

bool encodeToWav(const std::vector<float>& /*pcm*/,
                 uint32_t /*sampleRate*/,
                 uint16_t /*bitsPerSample*/,
                 std::vector<uint8_t>& /*wavData*/) {
    return false;
}

bool encodeAudioBuffer(const float* /*samples*/,
                       size_t /*numSamples*/,
                       uint32_t /*sampleRate*/,
                       uint16_t /*bitsPerSample*/,
                       const std::string& /*format*/,
                       std::vector<uint8_t>& /*output*/) {
    return false;
}

}  // namespace imp
}  // namespace ovms

#endif  // INTEL_MPI_AVAILABLE
