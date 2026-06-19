/**
 * Intel MPI - Internal Implementation Structures
 * 
 * NOT part of the public API. These are the C++ structs behind the
 * opaque handles declared in intel_mpi.h.
 *
 * Only intel_mpi.cpp and gst_loader.cpp should include this file.
 */

#ifndef IMP_MPI_IMPL_H
#define IMP_MPI_IMPL_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <chrono>

#ifdef _WIN32
// Windows + D3D11
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

// OpenVINO C++ API
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

// GStreamer (Windows: direct linking via import libs)
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/d3d11/gstd3d11.h>
#include <gst/pbutils/pbutils.h>

#else  // Linux

// OpenVINO C++ API
#include <openvino/openvino.hpp>

// GStreamer types needed for struct members (pointer-only use).
// Full headers are only included in gst_loader.cpp where the dlopen
// wrapper lives.  Using forward declarations here keeps intel_mpi.cpp
// free of any direct GStreamer symbol references.
typedef struct _GstElement   GstElement;
typedef struct _GstBus       GstBus;
typedef struct _GstMessage   GstMessage;

#endif  // _WIN32

// Public API types
#include "intel_mpi.h"

//////////////////////////////////////////////////////////////////////////////
// NV12 frame buffer (used internally for CPU-side NV12 data)
//////////////////////////////////////////////////////////////////////////////

struct imp_nv12_frame_t {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    int width  = 0;
    int height = 0;
    bool valid     = false;
    bool allocated = false;

    void allocate(int w, int h) {
        if (allocated && width == w && height == h) return;
        y_plane.resize(w * h);
        uv_plane.resize(w * (h / 2));
        width  = w;
        height = h;
        allocated = true;
        valid = false;
    }
};

//////////////////////////////////////////////////////////////////////////////
// Timing data (accumulated per-context for benchmarking)
//////////////////////////////////////////////////////////////////////////////

struct imp_timing_data_t {
    double context_create_ms           = 0;
    double video_open_ms               = 0;
    double encoder_open_ms             = 0;
    double total_decode_fullres_ms     = 0;
    double total_decode_small_ms       = 0;
    double total_decode_pull_fullres_ms = 0;
    double total_decode_pull_small_ms  = 0;
    double total_decode_copy_fullres_ms = 0;
    double total_decode_copy_small_ms  = 0;
    double total_inference_ms          = 0;
    double total_tensor_setup_ms       = 0;
    double total_infer_call_ms         = 0;
    double total_output_parse_ms       = 0;
    double total_async_start_ms        = 0;
    double total_async_wait_ms         = 0;
    double total_encode_ms             = 0;
    double total_display_ms            = 0;
    double total_overlay_ms            = 0;
    int    async_overlaps              = 0;
    int    frame_count                 = 0;
};

//////////////////////////////////////////////////////////////////////////////
// Context — wraps a compiled model + infer requests
//
// The public API creates this from an already-compiled OV model.
// The caller is responsible for:
//   1. Loading the model XML
//   2. Applying PrePostProcessor (NV12, resize, etc.)
//   3. Compiling the model for a device
//   4. Passing the compiled model here
//////////////////////////////////////////////////////////////////////////////

struct imp_context_s {
    // Device info (extracted from compiled model or created standalone)
    imp_device_type_t device_type   = IMP_DEVICE_GPU;
    std::string       device_name;         // e.g. "GPU.0"

    // Model dimension hints (for branch auto-deduction)
    // Set when context is created from a compiled model.
    // Used by imp_video_open when branch width/height are 0.
    int               model_w = 0;
    int               model_h = 0;

    bool              initialized   = false;
    std::string       last_error;
    imp_timing_data_t timing;
};

//////////////////////////////////////////////////////////////////////////////
// Tensor handle — wraps frame data for the C API
//
// Can hold either:
//   - NV12 frame data (y_plane + uv_plane, format = NV12)
//   - An ov::Tensor (for future use)
//
// NOTE: defined here (before imp_branch_info_t) because branches hold
//       a tensor_cache instance.
//////////////////////////////////////////////////////////////////////////////

struct imp_tensor_s {
    // NV12 frame data (when format is NV12)
    uint8_t*          y_data  = nullptr;   // Y plane pointer (not owned)
    uint8_t*          uv_data = nullptr;   // UV plane pointer (not owned)
    int               width   = 0;
    int               height  = 0;
    imp_pixel_format_t format = IMP_FORMAT_NV12;
    bool              valid   = false;

    // Optional ov::Tensor (for non-NV12 or GPU tensors)
    ov::Tensor        ov_tensor;
    imp_device_type_t device_type = IMP_DEVICE_CPU;
    std::string       device_name;
};

//////////////////////////////////////////////////////////////////////////////
// Branch info — one per output branch in the tee pipeline
//////////////////////////////////////////////////////////////////////////////

struct imp_branch_info_t {
    GstElement*      appsink = nullptr;
    int              width   = 0;
    int              height  = 0;
    std::string      name;
    imp_nv12_frame_t frame;             // pre-allocated frame buffer
    imp_tensor_s     tensor_cache;      // reusable tensor wrapper

    // Per-branch timing
    double total_decode_ms      = 0;
    double total_decode_pull_ms = 0;
    double total_decode_copy_ms = 0;
};

//////////////////////////////////////////////////////////////////////////////
// Video stream — GStreamer tee pipeline with N output branches
//////////////////////////////////////////////////////////////////////////////

struct imp_video_stream_s {
    GstElement* pipeline = nullptr;
    bool use_hw_decode   = false;

    // Dynamic branches (replaces fixed fullres + small appsinks)
    std::vector<imp_branch_info_t> branches;

    // Timing reference
    imp_timing_data_t* timing = nullptr;
};

//////////////////////////////////////////////////////////////////////////////
// Video encoder — GStreamer appsrc → encoder → mux → file
//////////////////////////////////////////////////////////////////////////////

struct imp_video_encoder_s {
    GstElement* pipeline = nullptr;
    GstElement* appsrc   = nullptr;
    int         width    = 0;
    int         height   = 0;
    int         fps_num  = 30;
    int         fps_den  = 1;
    bool        initialized    = false;
    bool        is_gpu_encoder = false;
    std::string encoder_name;
    int64_t     frame_count = 0;

    // Timing reference
    imp_timing_data_t* timing = nullptr;
};

////////////////////////////////////////////////////////////////////////////// 
// Video source configuration
//////////////////////////////////////////////////////////////////////////////

struct imp_video_source_s {
    imp_source_type_t type = IMP_SOURCE_FILE;
    std::string path;           // file path or URL
    std::string device;         // camera device id
    int width  = 0;
    int height = 0;
    int framerate = 0;
    std::string format;         // capture format
    // Network properties
    std::string transport;
    std::string username;
    std::string password;
    int timeout_ms = 0;
    bool low_latency = false;
};

//////////////////////////////////////////////////////////////////////////////
// Detection result (demo-level, not part of API)
//////////////////////////////////////////////////////////////////////////////

struct imp_detection_t {
    float confidence = 0.0f;
    int x_min = 0, y_min = 0, x_max = 0, y_max = 0;
};

//////////////////////////////////////////////////////////////////////////////
// Audio encoder — GStreamer pipeline for encoding audio samples to file
//////////////////////////////////////////////////////////////////////////////

struct imp_audio_encoder_s {
    GstElement* pipeline = nullptr;
    GstElement* appsrc   = nullptr;
    
    // Output configuration
    std::string output_path;
    std::string codec;                  // "mp3", "aac", "flac", "wav", "opus"
    uint32_t bitrate_kbps = 192;
    uint32_t sample_rate  = 44100;
    uint32_t channels     = 2;
    
    // Callback for completion notification
    imp_encode_callback_t callback = nullptr;
    void* user_data = nullptr;
    
    // State
    bool initialized = false;
    int64_t samples_written = 0;
    
    std::string last_error;
};

//////////////////////////////////////////////////////////////////////////////
// Audio stream — internal state for decode→encode pipeline
//////////////////////////////////////////////////////////////////////////////

struct imp_audio_stream_s {
    // Input
    std::string input_path;
    
    // Output
    std::string output_path;
    std::string output_codec;
    uint32_t    output_sample_rate  = 44100;
    uint32_t    output_channels     = 2;
    uint32_t    output_bitrate_kbps = 192;
    
    // Discovered info
    uint32_t    sample_rate  = 0;
    uint32_t    channels     = 0;
    double      duration_sec = 0.0;
    int64_t     num_samples  = 0;
    
    // Pipeline
    GstElement* pipeline = nullptr;
    
    // State
    bool        initialized  = false;
    bool        processed    = false;
    double      wall_time_sec = 0.0;
    
    std::string last_error;
};

#endif // IMP_MPI_IMPL_H
