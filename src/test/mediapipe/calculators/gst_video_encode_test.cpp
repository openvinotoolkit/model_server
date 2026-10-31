// Copyright (c) 2026 Intel Corporation
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

/**
 * Linux video encoder test — decode a few host NV12 frames, re-encode them to
 * H.264/MP4, finalize with EOS, then verify the output is readable by decoding
 * at least one frame back.
 *
 * Needs GStreamer + a test video; skips otherwise.
 */

#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "src/mpi/intel_mpi.h"

namespace {

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

TEST(GstVideoEncode, DecodeReencodeReadable) {
    const std::string video_path = envOr("VIDEO_PATH", "/ovms/recording_3_raw.avi");
    const std::string out_path = envOr("OUT_PATH", "/tmp/imp_encode_test.mp4");
    constexpr int kMaxFrames = 30;

    if (!fileExists(video_path))
        GTEST_SKIP() << "video not found: " << video_path;

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);

    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", video_path.c_str());

    imp_video_decode_opts_t vopts{};
    vopts.use_va_surface_memory = false;  // host NV12 planes for the encoder
    imp_video_stream_t* stream = nullptr;
    imp_status_t st = imp_video_open(&stream, src, ctx, &vopts);
    imp_video_source_destroy(src);
    if (st != IMP_OK) {
        imp_context_destroy(ctx);
        GTEST_SKIP() << "GStreamer decode unavailable: " << st;
    }

    uint32_t w = 0, h = 0;
    imp_video_get_info(stream, &w, &h, nullptr, nullptr);
    ASSERT_GT(w, 0u);
    ASSERT_GT(h, 0u);

    imp_video_encode_opts_t eopts{};
    eopts.codec = "h264";
    eopts.framerate = 30;
    eopts.output_path = out_path.c_str();

    imp_video_encoder_t* enc = nullptr;
    ASSERT_EQ(imp_video_encoder_create(&enc, w, h, ctx, &eopts), IMP_OK);

    int encoded = 0;
    for (; encoded < kMaxFrames;) {
        imp_tensor_t* tensor = nullptr;
        st = imp_video_read_frame(&tensor, stream, 0);
        if (st == IMP_ERROR_STREAM_END || st != IMP_OK || !tensor)
            break;
        EXPECT_EQ(imp_video_encoder_write(enc, tensor), IMP_OK);
        imp_tensor_release(tensor);
        encoded++;
    }
    imp_video_encoder_close(enc);  // EOS + finalize the file
    imp_video_close(stream);

    EXPECT_GT(encoded, 0);
    ASSERT_TRUE(fileExists(out_path)) << "encoder did not produce " << out_path;

    // Verify readable: reopen the encoded file and decode at least one frame.
    imp_video_source_t* src2 = nullptr;
    imp_video_source_create(&src2, IMP_SOURCE_FILE);
    imp_video_source_set(src2, "path", out_path.c_str());
    imp_video_decode_opts_t vopts2{};
    imp_video_stream_t* stream2 = nullptr;
    ASSERT_EQ(imp_video_open(&stream2, src2, ctx, &vopts2), IMP_OK)
        << "encoded file not readable";
    imp_video_source_destroy(src2);

    imp_tensor_t* t2 = nullptr;
    EXPECT_EQ(imp_video_read_frame(&t2, stream2, 0), IMP_OK) << "no frame from encoded file";
    if (t2)
        imp_tensor_release(t2);
    imp_video_close(stream2);

    imp_context_destroy(ctx);
    std::remove(out_path.c_str());
}

// GPU zero-copy encode: decode VA surfaces and push them straight into the VA
// encoder (no host copy), then verify the output is readable.
TEST(GstVideoEncode, GpuVaEncodeReadable) {
    const std::string video_path = envOr("VIDEO_PATH", "/ovms/recording_3_raw.avi");
    const std::string out_path = envOr("OUT_PATH_GPU", "/tmp/imp_encode_gpu.mp4");
    constexpr int kMaxFrames = 30;

    if (!fileExists(video_path))
        GTEST_SKIP() << "video not found: " << video_path;
    if (!imp_video_va_available())
        GTEST_SKIP() << "VA/GPU not available on this host";

    imp_context_t* ctx = nullptr;
    ASSERT_EQ(imp_context_create(&ctx, nullptr), IMP_OK);

    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, IMP_SOURCE_FILE);
    imp_video_source_set(src, "path", video_path.c_str());

    imp_video_decode_opts_t vopts{};
    vopts.use_va_surface_memory = true;  // VA surfaces for zero-copy encode
    imp_video_stream_t* stream = nullptr;
    imp_status_t st = imp_video_open(&stream, src, ctx, &vopts);
    imp_video_source_destroy(src);
    if (st != IMP_OK) {
        imp_context_destroy(ctx);
        GTEST_SKIP() << "GStreamer VA decode unavailable: " << st;
    }

    uint32_t w = 0, h = 0;
    imp_video_get_info(stream, &w, &h, nullptr, nullptr);
    ASSERT_GT(w, 0u);
    ASSERT_GT(h, 0u);

    imp_video_encode_opts_t eopts{};
    eopts.codec = "h264";
    eopts.framerate = 30;
    eopts.encode_device = "GPU";  // VA zero-copy encode path
    eopts.output_path = out_path.c_str();

    imp_video_encoder_t* enc = nullptr;
    ASSERT_EQ(imp_video_encoder_create(&enc, w, h, ctx, &eopts), IMP_OK)
        << "GPU encoder create failed: " << imp_context_get_error(ctx);

    int encoded = 0;
    for (; encoded < kMaxFrames;) {
        imp_tensor_t* tensor = nullptr;
        st = imp_video_read_frame(&tensor, stream, 0);
        if (st == IMP_ERROR_STREAM_END || st != IMP_OK || !tensor)
            break;
        EXPECT_EQ(imp_video_encoder_write(enc, tensor), IMP_OK);
        imp_tensor_release(tensor);  // encoder holds its own ref to the surface
        encoded++;
    }
    imp_video_encoder_close(enc);
    imp_video_close(stream);

    EXPECT_GT(encoded, 0);
    ASSERT_TRUE(fileExists(out_path)) << "GPU encoder did not produce " << out_path;

    // Verify readable.
    imp_video_source_t* src2 = nullptr;
    imp_video_source_create(&src2, IMP_SOURCE_FILE);
    imp_video_source_set(src2, "path", out_path.c_str());
    imp_video_decode_opts_t vopts2{};
    imp_video_stream_t* stream2 = nullptr;
    ASSERT_EQ(imp_video_open(&stream2, src2, ctx, &vopts2), IMP_OK)
        << "encoded file not readable";
    imp_video_source_destroy(src2);

    imp_tensor_t* t2 = nullptr;
    EXPECT_EQ(imp_video_read_frame(&t2, stream2, 0), IMP_OK) << "no frame from encoded file";
    if (t2)
        imp_tensor_release(t2);
    imp_video_close(stream2);

    imp_context_destroy(ctx);
    std::remove(out_path.c_str());
}

}  // namespace
