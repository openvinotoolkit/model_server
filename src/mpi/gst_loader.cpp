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
#include <fcntl.h>
#include <sys/stat.h>
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

    // libgstva-1.0 (VA surface extraction)
    // gst_va_buffer_get_surface returns VASurfaceID (uint32_t)
    uint32_t (*gst_va_buffer_get_surface)(GstBuffer*);

    // libva-drm / libva (VADisplay lifecycle)
    void* (*vaGetDisplayDRM)(int);                          // returns VADisplay
    int   (*vaInitialize)(void*, int*, int*);               // (VADisplay, &major, &minor)
    int   (*vaTerminate)(void*);                            // (VADisplay)
};

static GstFunctionTable s_fns       = {};
static bool s_available             = false;
static bool s_initialized           = false;
static bool s_va_available          = false;  // true when vapostproc element found
static bool s_va_surface_capable    = false;  // true when gst_va + libva dlopen succeeded
static void* s_va_display           = nullptr; // process-wide VADisplay (valid for lifetime of loader)
static int   s_drm_fd               = -1;      // /dev/dri/renderD128 fd (owned)

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
// Optional VA symbol loading (non-fatal — only enables surface-sharing path)
// ============================================================================

static void try_load_va_symbols() {
    void* gstva_h   = dlopen("libgstva-1.0.so.0",  RTLD_LAZY | RTLD_GLOBAL);
    void* vadrm_h   = dlopen("libva-drm.so.2",     RTLD_LAZY | RTLD_GLOBAL);
    void* va_h      = dlopen("libva.so.2",          RTLD_LAZY | RTLD_GLOBAL);

    if (!gstva_h || !vadrm_h || !va_h) return;

    auto* get_surf = reinterpret_cast<decltype(s_fns.gst_va_buffer_get_surface)>(
        dlsym(gstva_h, "gst_va_buffer_get_surface"));
    auto* get_disp = reinterpret_cast<decltype(s_fns.vaGetDisplayDRM)>(
        dlsym(vadrm_h, "vaGetDisplayDRM"));
    auto* va_init  = reinterpret_cast<decltype(s_fns.vaInitialize)>(
        dlsym(va_h, "vaInitialize"));
    auto* va_term  = reinterpret_cast<decltype(s_fns.vaTerminate)>(
        dlsym(va_h, "vaTerminate"));

    if (!get_surf || !get_disp || !va_init || !va_term) return;

    s_fns.gst_va_buffer_get_surface = get_surf;
    s_fns.vaGetDisplayDRM           = get_disp;
    s_fns.vaInitialize              = va_init;
    s_fns.vaTerminate               = va_term;

    // Open DRM device and initialise a VADisplay for the lifetime of this process.
    s_drm_fd = open("/dev/dri/renderD128", O_RDWR);
    if (s_drm_fd < 0) return;

    void* disp = get_disp(s_drm_fd);
    if (!disp) return;

    int maj = 0, min = 0;
    if (va_init(disp, &maj, &min) == 0 /* VA_STATUS_SUCCESS */) {
        s_va_display        = disp;
        s_va_surface_capable = true;
        std::cerr << "[GstLoader] VA surface sharing enabled (VA-API "
                  << maj << "." << min << ", DRM fd " << s_drm_fd << ").\n";
    } else {
        va_term(disp);
    }
}

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

    // Probe for VA-API support without loading the plugin in-process
    // (in-process factory probe causes heap corruption when dlopen'd glib
    //  conflicts with the system glib used by VA driver init).
    //
    // We check two conditions:
    //   1. The GStreamer VA plugin shared library is present.
    //   2. An Intel iHD VA driver is present in the DRI directory.
    //
    // Possible plugin locations (system install vs. custom prefix).
    static const char* va_plugin_paths[] = {
        "/opt/gstreamer/lib/gstreamer-1.0/libgstva.so",
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstva.so",
        nullptr,
    };
    static const char* va_driver_paths[] = {
        "/usr/lib/x86_64-linux-gnu/dri/iHD_drv_video.so",
        "/usr/lib/x86_64-linux-gnu/dri/i965_drv_video.so",
        nullptr,
    };

    auto file_exists = [](const char* p) {
        struct stat st;
        return stat(p, &st) == 0;
    };

    bool plugin_found = false;
    for (int i = 0; va_plugin_paths[i]; i++) {
        if (file_exists(va_plugin_paths[i])) { plugin_found = true; break; }
    }
    bool driver_found = false;
    for (int i = 0; va_driver_paths[i]; i++) {
        if (file_exists(va_driver_paths[i])) { driver_found = true; break; }
    }

    if (plugin_found && driver_found) {
        s_va_available = true;
        std::cerr << "[GstLoader] VA-API available — GPU decode enabled.\n";        // Try to also enable VA surface sharing (needs libgstva + libva).
        try_load_va_symbols();    } else {
        std::cerr << "[GstLoader] VA-API not available"
                  << (plugin_found ? "" : " (libgstva.so missing)")
                  << (driver_found ? "" : " (VA driver missing)")
                  << " — using CPU decode.\n";
    }

    return true;
}

bool gst_loader_available()    { return s_available; }
bool gst_loader_va_available() { return s_va_available; }
void* gst_loader_va_display()  { return s_va_display; }

void gst_loader_set_va_display(void* va_display) {
    // Allow the caller (e.g. after OV VAContext creation) to replace the
    // process-wide VADisplay so GStreamer VA surfaces are valid in OV's
    // GPU context.  The old display is intentionally NOT terminated here
    // because OV owns its own lifecycle; we simply stop using our probe copy.
    s_va_display = va_display;
    s_va_surface_capable = (va_display != nullptr);
}

// ============================================================================
// Internal: borrow-read one NV12 frame from an appsink (system-memory path).
//
// Zero-copy when the GstBuffer planes are packed (stride == width).
// The GstSample and mapped GstVideoFrame are kept alive in `branch` so the
// tensor's y_data/uv_data pointers remain valid until the NEXT call or close.
// Falls back to a compact memcpy when stride != width (uncommon).
// ============================================================================

static bool read_frame_borrow(imp_branch_info_t& branch,
                               double& total_ms,
                               double& pull_ms,
                               double& copy_ms) {
    const auto& f = s_fns;

    // Release the previous borrowed frame before pulling the next one.
    if (branch.live_vframe) {
        f.gst_video_frame_unmap(static_cast<GstVideoFrame*>(branch.live_vframe));
        delete static_cast<GstVideoFrame*>(branch.live_vframe);
        branch.live_vframe = nullptr;
    }
    if (branch.live_sample) {
        f.gst_sample_unref(static_cast<GstSample*>(branch.live_sample));
        branch.live_sample = nullptr;
    }

    auto t0 = chrono::high_resolution_clock::now();
    GstSample* sample = f.gst_app_sink_try_pull_sample(
        GST_APP_SINK(branch.appsink), GST_SECOND);
    auto t1 = chrono::high_resolution_clock::now();
    if (!sample) return false;
    pull_ms += chrono::duration<double, std::milli>(t1 - t0).count();

    GstBuffer* buffer = f.gst_sample_get_buffer(sample);
    GstCaps*   caps   = f.gst_sample_get_caps(sample);

    GstVideoInfo info;
    f.gst_video_info_from_caps(&info, caps);

    auto* vframe = new GstVideoFrame{};
    auto t2 = chrono::high_resolution_clock::now();

    if (!f.gst_video_frame_map(vframe, &info, buffer, GST_MAP_READ)) {
        delete vframe;
        f.gst_sample_unref(sample);
        return false;
    }

    uint8_t* y_ptr    = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(vframe, 0));
    int      y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(vframe, 0);
    uint8_t* uv_ptr   = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(vframe, 1));
    int      uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(vframe, 1);

    bool packed = (y_stride == info.width) && (uv_stride == info.width);

    if (packed) {
        // Zero-copy: tensor points directly into the mapped GstBuffer memory.
        branch.tensor_cache.y_data  = y_ptr;
        branch.tensor_cache.uv_data = uv_ptr;
        branch.live_sample = sample;
        branch.live_vframe = vframe;
    } else {
        // Fallback: compact copy (stride != width — uncommon, e.g. some cameras).
        branch.frame.allocate(info.width, info.height);
        for (int row = 0; row < info.height; row++)
            memcpy(branch.frame.y_plane.data()  + row * info.width,
                   y_ptr  + row * y_stride, info.width);
        for (int row = 0; row < info.height / 2; row++)
            memcpy(branch.frame.uv_plane.data() + row * info.width,
                   uv_ptr + row * uv_stride, info.width);
        branch.tensor_cache.y_data  = branch.frame.y_plane.data();
        branch.tensor_cache.uv_data = branch.frame.uv_plane.data();
        f.gst_video_frame_unmap(vframe);
        delete vframe;
        f.gst_sample_unref(sample);
    }

    branch.tensor_cache.width       = info.width;
    branch.tensor_cache.height      = info.height;
    branch.tensor_cache.valid       = true;
    branch.tensor_cache.memory_type = IMP_MEM_SYSTEM;

    auto t3 = chrono::high_resolution_clock::now();
    copy_ms  += chrono::duration<double, std::milli>(t3 - t2).count();
    total_ms += chrono::duration<double, std::milli>(t3 - t0).count();
    return true;
}

// ============================================================================
// Internal: read one NV12 frame as a VA surface (GPU surface-sharing path).
//
// The GstBuffer remains in VA memory — no DMA copy to system RAM.
// The VASurfaceID is extracted and stored in the tensor.
// The GstSample is kept alive until the next call or close.
// ============================================================================

static bool read_frame_va_surface(imp_branch_info_t& branch,
                                   void* va_display,
                                   double& total_ms,
                                   double& pull_ms,
                                   double& /*copy_ms*/) {
    const auto& f = s_fns;

    // Release previous VA surface sample.
    if (branch.live_va_sample) {
        f.gst_sample_unref(static_cast<GstSample*>(branch.live_va_sample));
        branch.live_va_sample = nullptr;
    }

    auto t0 = chrono::high_resolution_clock::now();
    GstSample* sample = f.gst_app_sink_try_pull_sample(
        GST_APP_SINK(branch.appsink), GST_SECOND);
    auto t1 = chrono::high_resolution_clock::now();
    if (!sample) return false;
    pull_ms += chrono::duration<double, std::milli>(t1 - t0).count();

    GstBuffer* buffer = f.gst_sample_get_buffer(sample);

    // Extract VASurfaceID directly — no system-memory map.
    uint32_t surface_id = f.gst_va_buffer_get_surface(buffer);

    branch.live_va_sample = sample;

    branch.tensor_cache.y_data       = nullptr;
    branch.tensor_cache.uv_data      = nullptr;
    branch.tensor_cache.width        = branch.width;
    branch.tensor_cache.height       = branch.height;
    branch.tensor_cache.valid        = true;
    branch.tensor_cache.memory_type  = IMP_MEM_VA_SURFACE;
    branch.tensor_cache.va_surface_id = surface_id;
    branch.tensor_cache.va_display    = va_display;

    auto t2 = chrono::high_resolution_clock::now();
    total_ms += chrono::duration<double, std::milli>(t2 - t0).count();
    return true;
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

    // Ensure GStreamer finds the custom plugin dir and VA driver before init.
    // setenv with overwrite=0 so user-set values take precedence.
    setenv("GST_PLUGIN_PATH", "/opt/gstreamer/lib/gstreamer-1.0", 0);
    setenv("LIBVA_DRIVER_NAME", "iHD", 0);

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

    // Choose decode pipeline:
    //   VA surface (GPU, zero-copy): decodebin ! vapostproc ! video/x-raw(memory:VAMemory) ! appsink
    //     - Decoded frame stays in GPU VA surface memory; no DMA copy.
    //     - Use imp_tensor_get_va_surface() + OV VAContext for inference.
    //   VA system (GPU, borrow):     decodebin ! vapostproc ! video/x-raw,format=NV12 ! appsink
    //     - GPU decode + borrow from GstBuffer: one unavoidable DMA copy VA→RAM,
    //       but our own memcpy is eliminated.
    //   CPU fallback (borrow):       decodebin ! videoconvert ! videoscale ! video/x-raw,format=NV12 ! appsink
    const bool want_va_surface = s_va_available && s_va_surface_capable
                                 && opts && opts->use_va_surface_memory;
    const bool use_va = s_va_available;

    auto make_branch_pipeline = [&](const std::string& loc,
                                     int w, int h,
                                     const std::string& sink_name,
                                     const std::string& sp) -> std::string {
        // VA surface path: keep frame in GPU memory (no DMA copy).
        if (want_va_surface) {
            std::string caps =
                "video/x-raw(memory:VAMemory),format=NV12,width=" + std::to_string(w) +
                ",height=" + std::to_string(h);
            return "filesrc location=\"" + loc + "\" ! "
                   "decodebin ! "
                   "vapostproc ! "
                   + caps + " ! "
                   "appsink name=" + sink_name + " " + sp;
        }
        // System-memory paths (borrow or CPU decode).
        std::string caps =
            "video/x-raw,format=NV12,width=" + std::to_string(w) +
            ",height=" + std::to_string(h);
        if (use_va) {
            return "filesrc location=\"" + loc + "\" ! "
                   "decodebin ! "
                   "vapostproc ! "
                   + caps + " ! "
                   "appsink name=" + sink_name + " " + sp;
        } else {
            return "filesrc location=\"" + loc + "\" ! "
                   "decodebin ! "
                   "videoconvert ! "
                   "videoscale ! "
                   + caps + " ! "
                   "appsink name=" + sink_name + " " + sp;
        }
    };

    if (resolved.size() == 1) {
        const auto& rb = resolved[0];
        pipeline_str = make_branch_pipeline(path, rb.width, rb.height,
                                             "branch_0", sink_props);
    } else {
        // Multi-branch via tee — VA path uses vapostproc per branch
        std::string convert_scale;
        if (want_va_surface) {
            convert_scale = "vapostproc";
        } else if (use_va) {
            convert_scale = "vapostproc";
        } else {
            convert_scale = "videoconvert ! videoscale";
        }
        pipeline_str =
            "filesrc location=\"" + path + "\" ! "
            "decodebin ! "
            "tee name=t";

        for (size_t i = 0; i < resolved.size(); i++) {
            const auto& rb = resolved[i];
            std::string sink_name = "branch_" + std::to_string(i);
            std::string mem_caps = want_va_surface
                ? "video/x-raw(memory:VAMemory),format=NV12,width=" + std::to_string(rb.width) + ",height=" + std::to_string(rb.height)
                : "video/x-raw,format=NV12,width=" + std::to_string(rb.width) + ",height=" + std::to_string(rb.height);
            pipeline_str +=
                " t. ! queue ! " + convert_scale + " ! " +
                mem_caps + " ! "
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

    s->use_hw_decode         = use_va;
    s->use_va_surface_memory = want_va_surface;
    s->va_display            = want_va_surface ? s_va_display : nullptr;

    auto end = chrono::high_resolution_clock::now();
    ctx->timing.video_open_ms =
        chrono::duration<double, std::milli>(end - start).count();

    const char* decode_mode = want_va_surface ? "VA-API/GPU surface"
                            : use_va          ? "VA-API/GPU"
                            :                   "CPU";
    for (size_t i = 0; i < s->branches.size(); i++) {
        const auto& b = s->branches[i];
        std::cout << "[linux_video_open] Branch " << i
                  << " [" << b.name << "]: "
                  << b.width << "x" << b.height
                  << " NV12 (" << decode_mode << ")\n";
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

    bool ok;
    if (stream->use_va_surface_memory) {
        ok = read_frame_va_surface(
            branch, stream->va_display,
            branch.total_decode_ms,
            branch.total_decode_pull_ms,
            branch.total_decode_copy_ms);
    } else {
        ok = read_frame_borrow(
            branch,
            branch.total_decode_ms,
            branch.total_decode_pull_ms,
            branch.total_decode_copy_ms);
    }

    if (!ok) return IMP_ERROR_STREAM_END;

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

    // Release any live borrowed/VA frames BEFORE stopping the pipeline —
    // the buffers must be returned to GStreamer's pool first.
    for (auto& b : stream->branches) {
        if (b.live_vframe) {
            f.gst_video_frame_unmap(static_cast<GstVideoFrame*>(b.live_vframe));
            delete static_cast<GstVideoFrame*>(b.live_vframe);
            b.live_vframe = nullptr;
        }
        if (b.live_sample) {
            f.gst_sample_unref(static_cast<GstSample*>(b.live_sample));
            b.live_sample = nullptr;
        }
        if (b.live_va_sample) {
            f.gst_sample_unref(static_cast<GstSample*>(b.live_va_sample));
            b.live_va_sample = nullptr;
        }
    }

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
