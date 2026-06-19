// Copyright (c) 2025 Intel Corporation
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
 * GStreamer runtime loader — Linux implementation.
 *
 * This is the ONLY translation unit that may include GStreamer headers.
 * All gst_* calls go through function pointers resolved via dlopen so
 * that the binary has no DT_NEEDED entry for GStreamer and can start on
 * systems where GStreamer is not installed.
 *
 * TODO (Phase 2): verify graceful degradation in release image without
 * GStreamer runtime packages.
 */

#ifndef _WIN32

// ---- GStreamer headers (compile-time type definitions and macros) ----------
// These are only needed here because we use GstVideoFrame / GstVideoInfo by
// value (for GST_VIDEO_FRAME_PLANE_DATA / _STRIDE macros which are struct
// field accesses, not function calls).
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

// ---- standard headers -------------------------------------------------------
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>

// ---- Intel MPI public API ---------------------------------------------------
#include "intel_mpi.h"

// ---- Internal structs (needed for branch / stream layout) ------------------
// We include only what's needed for the Linux video path; Windows-only
// sections (D3D11, etc.) are guarded in imp_mpi_impl.h.
#include "imp_mpi_impl.h"

namespace chrono = std::chrono;

// ============================================================================
// Function pointer table
// ============================================================================

struct GstFunctionTable {
    // gstreamer-1.0
    void (*gst_init)(int*, char***);
    GstElement* (*gst_parse_launch)(const gchar*, GError**);
    GstStateChangeReturn (*gst_element_set_state)(GstElement*, GstState);
    GstStateChangeReturn (*gst_element_get_state)(GstElement*, GstState*,
                                                   GstState*, GstClockTime);
    GstElement* (*gst_bin_get_by_name)(GstBin*, const gchar*);
    GstPad* (*gst_element_get_static_pad)(GstElement*, const gchar*);
    GstCaps* (*gst_pad_get_current_caps)(GstPad*);
    GstStructure* (*gst_caps_get_structure)(const GstCaps*, guint);
    gboolean (*gst_structure_get_int)(const GstStructure*, const gchar*, gint*);
    void (*gst_caps_unref)(GstCaps*);
    void (*gst_object_unref)(gpointer);
    GstBus* (*gst_element_get_bus)(GstElement*);
    gboolean (*gst_element_send_event)(GstElement*, GstEvent*);
    GstEvent* (*gst_event_new_eos)(void);
    GstMessage* (*gst_bus_timed_pop_filtered)(GstBus*, GstClockTime,
                                               GstMessageType);
    void (*gst_message_unref)(GstMessage*);

    // gstapp-1.0
    GstSample* (*gst_app_sink_try_pull_sample)(GstAppSink*, GstClockTime);
    GstBuffer* (*gst_sample_get_buffer)(GstSample*);
    GstCaps* (*gst_sample_get_caps)(GstSample*);
    void (*gst_sample_unref)(GstSample*);

    // gstvideo-1.0
    gboolean (*gst_video_info_from_caps)(GstVideoInfo*, const GstCaps*);
    gboolean (*gst_video_frame_map)(GstVideoFrame*, GstVideoInfo*,
                                     GstBuffer*, GstMapFlags);
    void (*gst_video_frame_unmap)(GstVideoFrame*);

    // glib-2.0
    void (*g_error_free)(GError*);
};

static GstFunctionTable s_fns = {};
static bool s_available       = false;
static bool s_initialized     = false;

// ============================================================================
// Symbol loading helpers
// ============================================================================

#define LOAD_SYM(handle, sym)                                           \
    do {                                                                \
        s_fns.sym = reinterpret_cast<decltype(s_fns.sym)>(             \
            dlsym((handle), #sym));                                     \
        if (!s_fns.sym) {                                               \
            std::cerr << "[GstLoader] Missing symbol: " #sym "\n";     \
            return false;                                               \
        }                                                               \
    } while (0)

static bool load_symbols(void* gst_h, void* app_h,
                         void* vid_h, void* glib_h) {
    LOAD_SYM(gst_h,  gst_init);
    LOAD_SYM(gst_h,  gst_parse_launch);
    LOAD_SYM(gst_h,  gst_element_set_state);
    LOAD_SYM(gst_h,  gst_element_get_state);
    LOAD_SYM(gst_h,  gst_bin_get_by_name);
    LOAD_SYM(gst_h,  gst_element_get_static_pad);
    LOAD_SYM(gst_h,  gst_pad_get_current_caps);
    LOAD_SYM(gst_h,  gst_caps_get_structure);
    LOAD_SYM(gst_h,  gst_structure_get_int);
    LOAD_SYM(gst_h,  gst_caps_unref);
    LOAD_SYM(gst_h,  gst_object_unref);
    LOAD_SYM(gst_h,  gst_element_get_bus);
    LOAD_SYM(gst_h,  gst_element_send_event);
    LOAD_SYM(gst_h,  gst_event_new_eos);
    LOAD_SYM(gst_h,  gst_bus_timed_pop_filtered);
    LOAD_SYM(gst_h,  gst_message_unref);

    LOAD_SYM(app_h,  gst_app_sink_try_pull_sample);
    LOAD_SYM(app_h,  gst_sample_get_buffer);
    LOAD_SYM(app_h,  gst_sample_get_caps);
    LOAD_SYM(app_h,  gst_sample_unref);

    LOAD_SYM(vid_h,  gst_video_info_from_caps);
    LOAD_SYM(vid_h,  gst_video_frame_map);
    LOAD_SYM(vid_h,  gst_video_frame_unmap);

    LOAD_SYM(glib_h, g_error_free);

    return true;
}

#undef LOAD_SYM

// ============================================================================
// Public loader API
// ============================================================================

bool gst_loader_init() {
    if (s_initialized) return s_available;
    s_initialized = true;

    void* gst_h  = dlopen("libgstreamer-1.0.so.0",  RTLD_LAZY | RTLD_GLOBAL);
    void* app_h  = dlopen("libgstapp-1.0.so.0",     RTLD_LAZY | RTLD_GLOBAL);
    void* vid_h  = dlopen("libgstvideo-1.0.so.0",   RTLD_LAZY | RTLD_GLOBAL);
    void* glib_h = dlopen("libglib-2.0.so.0",       RTLD_LAZY | RTLD_GLOBAL);

    if (!gst_h || !app_h || !vid_h || !glib_h) {
        std::cerr << "[GstLoader] GStreamer not available: " << dlerror() << "\n"
                  << "[GstLoader] Install gstreamer1.0-* packages to enable "
                     "media decode.\n";
        return false;
    }

    if (!load_symbols(gst_h, app_h, vid_h, glib_h)) {
        return false;
    }

    s_available = true;
    std::cerr << "[GstLoader] GStreamer loaded successfully.\n";
    return true;
}

bool gst_loader_available() { return s_available; }

// ============================================================================
// Internal: read one NV12 frame from an appsink
// ============================================================================

static bool read_nv12_from_appsink(GstElement* appsink,
                                    imp_nv12_frame_t* frame,
                                    double& total_ms,
                                    double& pull_ms,
                                    double& copy_ms) {
    const auto& f = s_fns;
    auto decode_start = chrono::high_resolution_clock::now();

    auto ps = chrono::high_resolution_clock::now();
    GstSample* sample = f.gst_app_sink_try_pull_sample(
        GST_APP_SINK(appsink), GST_SECOND);
    if (!sample) return false;
    pull_ms += chrono::duration<double, std::milli>(
        chrono::high_resolution_clock::now() - ps).count();

    GstBuffer* buffer = f.gst_sample_get_buffer(sample);
    GstCaps*   caps   = f.gst_sample_get_caps(sample);

    GstVideoInfo info;
    f.gst_video_info_from_caps(&info, caps);

    frame->allocate(info.width, info.height);
    frame->valid = false;

    auto cs = chrono::high_resolution_clock::now();

    GstVideoFrame vframe;
    if (f.gst_video_frame_map(&vframe, &info, buffer, GST_MAP_READ)) {
        uint8_t* y_data   = static_cast<uint8_t*>(
            GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0));
        int      y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);

        if (y_stride == frame->width) {
            memcpy(frame->y_plane.data(), y_data,
                   frame->width * frame->height);
        } else {
            for (int row = 0; row < frame->height; row++)
                memcpy(frame->y_plane.data() + row * frame->width,
                       y_data + row * y_stride, frame->width);
        }

        uint8_t* uv_data   = static_cast<uint8_t*>(
            GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1));
        int      uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);

        if (uv_stride == frame->width) {
            memcpy(frame->uv_plane.data(), uv_data,
                   frame->width * (frame->height / 2));
        } else {
            for (int row = 0; row < frame->height / 2; row++)
                memcpy(frame->uv_plane.data() + row * frame->width,
                       uv_data + row * uv_stride, frame->width);
        }

        frame->valid = true;
        f.gst_video_frame_unmap(&vframe);
    }

    f.gst_sample_unref(sample);

    copy_ms  += chrono::duration<double, std::milli>(
        chrono::high_resolution_clock::now() - cs).count();
    total_ms += chrono::duration<double, std::milli>(
        chrono::high_resolution_clock::now() - decode_start).count();

    return frame->valid;
}

// ============================================================================
// Public: linux_video_open
// ============================================================================

imp_status_t linux_video_open(imp_video_stream_t** stream,
                               imp_video_source_t* source,
                               imp_context_t* ctx,
                               const imp_video_decode_opts_t* opts) {
    if (!stream || !source || !ctx) return IMP_ERROR_INVALID_ARGUMENT;

    if (!gst_loader_available()) {
        ctx->last_error = "GStreamer not available; install gstreamer1.0-* "
                          "packages and retry.";
        return IMP_ERROR_DEVICE_NOT_AVAILABLE;
    }

    const auto& f = s_fns;
    auto start = chrono::high_resolution_clock::now();

    f.gst_init(nullptr, nullptr);

    auto* s   = new imp_video_stream_s();
    s->timing = &ctx->timing;

    int src_w = (source->width  > 0) ? source->width  : 1280;
    int src_h = (source->height > 0) ? source->height : 720;

    bool is_file = (source->type == IMP_SOURCE_FILE);

    // Build branch list (single branch only for initial Linux impl)
    struct resolved_branch { int width; int height; std::string name; };
    std::vector<resolved_branch> resolved;

    if (opts && opts->branches && opts->branch_count > 0) {
        for (uint32_t i = 0; i < opts->branch_count; i++) {
            resolved_branch rb;
            int bw = static_cast<int>(opts->branches[i].width);
            int bh = static_cast<int>(opts->branches[i].height);
            if (bw > 0 && bh > 0) {
                rb.width  = bw;
                rb.height = bh;
            } else if (ctx->model_w > 0 && ctx->model_h > 0) {
                rb.width  = ctx->model_w;
                rb.height = ctx->model_h;
            } else {
                rb.width  = src_w;
                rb.height = src_h;
            }
            rb.name = opts->branches[i].name
                      ? opts->branches[i].name
                      : ("branch_" + std::to_string(i));
            resolved.push_back(rb);
        }
    } else {
        resolved.push_back({src_w, src_h, "default"});
    }

    // appsink properties
    std::string sink_props = is_file
        ? "emit-signals=false sync=false max-buffers=0 drop=false"
        : "emit-signals=false sync=false max-buffers=2 drop=true";

    if (is_file && opts && opts->sync_appsink) {
        sink_props = "emit-signals=false sync=false max-buffers=1 drop=false";
    }

    // Linux CPU decode pipeline:
    //   filesrc | decodebin | videoconvert | videoscale | caps | appsink
    //
    // TODO: VA-API GPU path (deferred — hardware not available yet):
    //   filesrc | decodebin | vapostproc | caps | appsink
    std::string pipeline_str;

    if (!is_file) {
        // Camera / live source not yet implemented on Linux
        ctx->last_error = "Live camera source not yet supported on Linux";
        delete s;
        return IMP_ERROR_UNSUPPORTED_FORMAT;
    }

    // Normalize path (backslashes break gst_parse_launch)
    std::string path = source->path;
    std::replace(path.begin(), path.end(), '\\', '/');

    if (resolved.size() == 1) {
        const auto& rb = resolved[0];
        pipeline_str =
            "filesrc location=\"" + path + "\" ! "
            "decodebin ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,format=NV12,"
            "width=" + std::to_string(rb.width) +
            ",height=" + std::to_string(rb.height) + " ! "
            "appsink name=branch_0 " + sink_props;
    } else {
        // Multi-branch via tee
        pipeline_str =
            "filesrc location=\"" + path + "\" ! "
            "decodebin ! "
            "videoconvert ! tee name=t";

        for (size_t i = 0; i < resolved.size(); i++) {
            const auto& rb = resolved[i];
            std::string sink_name = "branch_" + std::to_string(i);
            pipeline_str +=
                " t. ! queue ! "
                "videoscale ! "
                "video/x-raw,format=NV12,"
                "width=" + std::to_string(rb.width) +
                ",height=" + std::to_string(rb.height) + " ! "
                "appsink name=" + sink_name + " " + sink_props;
        }
    }

    std::cout << "[linux_video_open] Pipeline: " << pipeline_str << "\n";

    GError* error = nullptr;
    s->pipeline = f.gst_parse_launch(pipeline_str.c_str(), &error);
    if (error || !s->pipeline) {
        ctx->last_error = error ? error->message : "pipeline creation failed";
        if (error) f.g_error_free(error);
        delete s;
        return IMP_ERROR_DECODE_FAILED;
    }

    s->branches.resize(resolved.size());
    for (size_t i = 0; i < resolved.size(); i++) {
        std::string sink_name = "branch_" + std::to_string(i);
        s->branches[i].appsink = f.gst_bin_get_by_name(
            GST_BIN(s->pipeline), sink_name.c_str());
        if (!s->branches[i].appsink) {
            ctx->last_error = "failed to get appsink: " + sink_name;
            f.gst_object_unref(s->pipeline);
            delete s;
            return IMP_ERROR_DECODE_FAILED;
        }
        s->branches[i].width  = resolved[i].width;
        s->branches[i].height = resolved[i].height;
        s->branches[i].name   = resolved[i].name;
        s->branches[i].frame.allocate(resolved[i].width, resolved[i].height);
    }

    f.gst_element_set_state(s->pipeline, GST_STATE_PLAYING);
    GstStateChangeReturn ret = f.gst_element_get_state(
        s->pipeline, nullptr, nullptr, 5 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        ctx->last_error = "pipeline failed to start";
        for (auto& b : s->branches) {
            if (b.appsink) f.gst_object_unref(b.appsink);
        }
        f.gst_object_unref(s->pipeline);
        delete s;
        return IMP_ERROR_DECODE_FAILED;
    }

    // Read negotiated caps from first appsink to get actual dimensions
    GstPad* pad = f.gst_element_get_static_pad(s->branches[0].appsink, "sink");
    if (pad) {
        GstCaps* caps = f.gst_pad_get_current_caps(pad);
        if (caps) {
            GstStructure* st = f.gst_caps_get_structure(caps, 0);
            int actual_w = 0, actual_h = 0;
            f.gst_structure_get_int(st, "width",  &actual_w);
            f.gst_structure_get_int(st, "height", &actual_h);
            if (actual_w > 0) s->branches[0].width  = actual_w;
            if (actual_h > 0) s->branches[0].height = actual_h;
            f.gst_caps_unref(caps);
        }
        f.gst_object_unref(pad);
    }

    s->use_hw_decode = false;  // CPU decode

    auto end = chrono::high_resolution_clock::now();
    ctx->timing.video_open_ms =
        chrono::duration<double, std::milli>(end - start).count();

    for (size_t i = 0; i < s->branches.size(); i++) {
        const auto& b = s->branches[i];
        std::cout << "[linux_video_open] Branch " << i
                  << " [" << b.name << "]: "
                  << b.width << "x" << b.height << " NV12 (CPU)\n";
    }

    *stream = s;
    return IMP_OK;
}

// ============================================================================
// Public: linux_video_read_frame
// ============================================================================

imp_status_t linux_video_read_frame(imp_tensor_t** tensor,
                                     imp_video_stream_t* stream,
                                     uint32_t branch_index) {
    if (!stream || branch_index >= static_cast<uint32_t>(stream->branches.size()))
        return IMP_ERROR_INVALID_ARGUMENT;

    auto& branch = stream->branches[branch_index];

    bool ok = read_nv12_from_appsink(
        branch.appsink, &branch.frame,
        branch.total_decode_ms,
        branch.total_decode_pull_ms,
        branch.total_decode_copy_ms);

    if (!ok) return IMP_ERROR_STREAM_END;

    branch.tensor_cache.y_data      = branch.frame.y_plane.data();
    branch.tensor_cache.uv_data     = branch.frame.uv_plane.data();
    branch.tensor_cache.width       = branch.frame.width;
    branch.tensor_cache.height      = branch.frame.height;
    branch.tensor_cache.format      = IMP_FORMAT_NV12;
    branch.tensor_cache.valid       = true;
    branch.tensor_cache.device_type = IMP_DEVICE_CPU;

    if (tensor) *tensor = &branch.tensor_cache;
    return IMP_OK;
}

// ============================================================================
// Public: linux_video_close
// ============================================================================

void linux_video_close(imp_video_stream_t* stream) {
    if (!stream) return;
    const auto& f = s_fns;
    if (stream->pipeline) {
        f.gst_element_set_state(stream->pipeline, GST_STATE_NULL);
        for (auto& b : stream->branches) {
            if (b.appsink) f.gst_object_unref(b.appsink);
        }
        f.gst_object_unref(stream->pipeline);
    }
    delete stream;
}

#endif  // !_WIN32
