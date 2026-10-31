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

#pragma once

/**
 * @file imp_audio_utils.hpp
 * @brief Intel MPI audio utilities — thin wrappers around the Intel MPI
 *        decode/encode API for use inside OVMS audio calculators.
 *
 * When INTEL_MPI_AVAILABLE is defined (Windows builds with GStreamer),
 * these helpers provide GStreamer-backed audio decode that supports a
 * wider range of formats (ogg, flac, aac, opus, …) beyond the basic
 * WAV/MP3 support offered by dr_libs.
 *
 * On Linux (or when GStreamer is not present), the functions return false
 * so the caller can fall back to the existing dr_libs path.
 */

#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace ovms {
namespace imp {

/**
 * Check whether Intel MPI audio decode is available at runtime.
 * Returns true only on Windows builds compiled with INTEL_MPI_AVAILABLE
 * AND where GStreamer is actually initialised.
 */
bool isAvailable();

/**
 * Try to decode an audio buffer (any format GStreamer supports) into
 * mono float PCM samples at the requested sample rate.
 *
 * @param audioData   Raw encoded audio bytes (WAV, MP3, OGG, FLAC, …)
 * @param sampleRate  Desired output sample rate (e.g. 16000)
 * @param[out] pcm    Output float samples, normalised to [-1, 1]
 * @return true on success, false if MPI is unavailable or decode fails.
 */
bool decodeAudioBuffer(const std::string_view& audioData,
                       uint32_t sampleRate,
                       std::vector<float>& pcm);

/**
 * Encode float PCM samples to a WAV buffer in memory.
 *
 * @param pcm          Input float samples (mono, normalised [-1,1])
 * @param sampleRate   Sample rate (e.g. 16000)
 * @param bitsPerSample  Bits per sample for output (16 or 32)
 * @param[out] wavData  Output WAV bytes
 * @return true on success.
 */
bool encodeToWav(const std::vector<float>& pcm,
                 uint32_t sampleRate,
                 uint16_t bitsPerSample,
                 std::vector<uint8_t>& wavData);

/**
 * Encode float PCM samples to the specified audio format.
 *
 * Supported formats: "wav", "mp3", "flac", "opus", "aac", "pcm".
 * On Windows, non-WAV formats are encoded via the Intel MPI / GStreamer
 * pipeline.  On Linux (stub), only "wav" and "pcm" are supported.
 *
 * @param samples        Raw float PCM (mono)
 * @param numSamples     Number of float values
 * @param sampleRate     Sample rate (e.g. 16000)
 * @param bitsPerSample  Bits per sample for WAV output (16 or 32)
 * @param format         Target format string ("wav","mp3","flac",...)
 * @param[out] output    Encoded bytes
 * @return true on success.
 */
bool encodeAudioBuffer(const float* samples,
                       size_t numSamples,
                       uint32_t sampleRate,
                       uint16_t bitsPerSample,
                       const std::string& format,
                       std::vector<uint8_t>& output);

}  // namespace imp
}  // namespace ovms
