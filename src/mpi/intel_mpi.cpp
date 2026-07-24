/**
 * Intel MPI - API Implementation (v8 — Layer 1 Media I/O)
 *
 * Implements the C-compatible functions declared in intel_mpi.h.
 * Uses the internal C++ structs from imp_mpi_impl.h.
 *
 * This is a PURE MEDIA layer:
 *   - Context = device handle (extracts GPU device from model, or standalone)
 *   - Video decode = N-branch tee pipeline (caller defines branches)
 *   - Tensor = NV12 frame wrapper returned by imp_video_read_frame
 *   - Encoder = accepts imp_tensor_t* directly
 *
 * NO inference logic lives here. Inference helpers belong in the demo/OVMS layer.
 *
 * Platform notes:
 *   Windows — direct GStreamer linking via import libs; D3D11 HW pipeline.
 *   Linux   — GStreamer loaded at runtime via dlopen (gst_loader.cpp);
 *             CPU decode pipeline only (VA-API GPU path deferred).
 *             Image / audio encode-decode functions are Windows-only for now
 *             and return IMP_ERROR_INTERNAL on Linux.
 */

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#ifndef GST_USE_UNSTABLE_API
#define GST_USE_UNSTABLE_API
#endif
#endif  // _WIN32

#include "imp_mpi_impl.h"  // same directory

#ifndef _WIN32
#include "gst_loader.h"
#endif

#include <iostream>
#include <algorithm>
#include <chrono>

namespace chrono = std::chrono;

//////////////////////////////////////////////////////////////////////////////
// Version
//////////////////////////////////////////////////////////////////////////////

void imp_get_version(int* major, int* minor, int* patch) {
    if (major) *major = IMP_VERSION_MAJOR;
    if (minor) *minor = IMP_VERSION_MINOR;
    if (patch) *patch = IMP_VERSION_PATCH;
}

//////////////////////////////////////////////////////////////////////////////
// Context — device handle only, no inference
//////////////////////////////////////////////////////////////////////////////

imp_status_t imp_context_create(imp_context_t** ctx,
                                ov_compiled_model_t* compiled_model) {
    if (!ctx) return IMP_ERROR_INVALID_ARGUMENT;

    // compiled_model is a C++ ov::CompiledModel* cast through the C opaque pointer.
    // On Linux / CPU-only mode, nullptr is valid — creates a standalone context
    // with no model dimension hints.
    auto* cm = reinterpret_cast<ov::CompiledModel*>(compiled_model);
#ifdef _WIN32
    if (!cm) return IMP_ERROR_INVALID_ARGUMENT;
#endif

    auto start = chrono::high_resolution_clock::now();

    auto* c = new imp_context_s();

    // Extract model input dimensions for branch auto-deduction.
    // After NV12 PPP the Y input is [1, H, W, 1].
    if (cm) {
        // Extract model input dimensions for branch auto-deduction.
        // After NV12 PPP the Y input is [1, H, W, 1].
        auto inputs = cm->inputs();
        if (!inputs.empty()) {
            auto shape = inputs[0].get_shape();
            if (shape.size() == 4) {
                c->model_h = static_cast<int>(shape[1]);
                c->model_w = static_cast<int>(shape[2]);
            }
        }
    }

    c->device_type = IMP_DEVICE_CPU;
    c->device_name = cm ? "GPU" : "CPU";
    c->initialized = true;

    auto end = chrono::high_resolution_clock::now();
    c->timing.context_create_ms = chrono::duration<double, std::milli>(end - start).count();

    *ctx = c;
    return IMP_OK;
}

imp_status_t imp_context_create_from_remote(imp_context_t** ctx,
                                            ov_remote_context_t* remote_ctx,
                                            imp_context_type_t type) {
    (void)ctx; (void)remote_ctx; (void)type;
    return IMP_ERROR_INTERNAL; // Not yet implemented
}

imp_status_t imp_context_get_native(imp_context_t* ctx,
                                    imp_context_type_t* type,
                                    void** native_handle) {
    (void)ctx; (void)type; (void)native_handle;
    return IMP_ERROR_INTERNAL; // Not yet implemented
}

imp_status_t imp_context_get_ov_remote(imp_context_t* ctx,
                                       ov_remote_context_t** remote_ctx) {
    (void)ctx; (void)remote_ctx;
    return IMP_ERROR_INTERNAL; // Not yet implemented
}

imp_status_t imp_context_get_device_type(imp_context_t* ctx,
                                         imp_device_type_t* device_type) {
    if (!ctx || !device_type) return IMP_ERROR_INVALID_ARGUMENT;
    *device_type = ctx->device_type;
    return IMP_OK;
}

imp_status_t imp_context_get_device_name(imp_context_t* ctx,
                                         const char** device_name) {
    if (!ctx || !device_name) return IMP_ERROR_INVALID_ARGUMENT;
    *device_name = ctx->device_name.c_str();
    return IMP_OK;
}

void imp_context_destroy(imp_context_t* ctx) {
    delete ctx;
}

const char* imp_context_get_error(imp_context_t* ctx) {
    if (!ctx) return "null context";
    return ctx->last_error.c_str();
}

//////////////////////////////////////////////////////////////////////////////
// Video Source Configuration
//////////////////////////////////////////////////////////////////////////////

imp_status_t imp_video_source_create(imp_video_source_t** source,
                                     imp_source_type_t type) {
    if (!source) return IMP_ERROR_INVALID_ARGUMENT;
    auto* s = new imp_video_source_s();
    s->type = type;
    *source = s;
    return IMP_OK;
}

imp_status_t imp_video_source_set(imp_video_source_t* source,
                                  const char* key,
                                  const char* value) {
    if (!source || !key || !value) return IMP_ERROR_INVALID_ARGUMENT;

    std::string k(key), v(value);
    if (k == "path" || k == "url")  source->path = v;
    else if (k == "device")         source->device = v;
    else if (k == "width")          source->width = std::stoi(v); // no error handling FIXME stoi
    else if (k == "height")         source->height = std::stoi(v);
    else if (k == "framerate")      source->framerate = std::stoi(v);
    else if (k == "format")         source->format = v;
    else if (k == "transport")      source->transport = v;
    else if (k == "username")       source->username = v;
    else if (k == "password")       source->password = v;
    else if (k == "timeout")        source->timeout_ms = std::stoi(v);
    else if (k == "low_latency")    source->low_latency = (v == "1" || v == "true");
    else return IMP_ERROR_INVALID_ARGUMENT;

    return IMP_OK;
}

void imp_video_source_destroy(imp_video_source_t* source) {
    delete source;
}

// ============================================================================
// Windows-only GStreamer video / audio / image pipeline implementations.
// Linux equivalents live in gst_loader.cpp (dlopen-based).
// ============================================================================
#ifdef _WIN32

//////////////////////////////////////////////////////////////////////////////
// Internal: read NV12 from appsink into frame buffer
//////////////////////////////////////////////////////////////////////////////

static bool read_nv12_from_appsink(GstElement* appsink, imp_nv12_frame_t* frame,
                                   double& total_ms, double& pull_ms, double& copy_ms) {
    auto decode_start = chrono::high_resolution_clock::now();

    auto ps = chrono::high_resolution_clock::now();
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), GST_SECOND);
    if (!sample) return false;
    pull_ms += chrono::duration<double, std::milli>(
        chrono::high_resolution_clock::now() - ps).count();

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps*   caps   = gst_sample_get_caps(sample);

    GstVideoInfo info;
    gst_video_info_from_caps(&info, caps);

    frame->allocate(info.width, info.height);
    frame->valid = false;

    auto cs = chrono::high_resolution_clock::now();

    GstVideoFrame vframe;
    if (gst_video_frame_map(&vframe, &info, buffer, GST_MAP_READ)) {
        uint8_t* y_data  = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
        int      y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);

        if (y_stride == frame->width) {
            memcpy(frame->y_plane.data(), y_data, frame->width * frame->height);
        } else {
            for (int row = 0; row < frame->height; row++)
                memcpy(frame->y_plane.data() + row * frame->width,
                       y_data + row * y_stride, frame->width);
        }

        uint8_t* uv_data  = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1);
        int      uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);

        if (uv_stride == frame->width) {
            memcpy(frame->uv_plane.data(), uv_data, frame->width * (frame->height / 2));
        } else {
            for (int row = 0; row < frame->height / 2; row++)
                memcpy(frame->uv_plane.data() + row * frame->width,
                       uv_data + row * uv_stride, frame->width);
        }

        frame->valid = true;
        gst_video_frame_unmap(&vframe);
    }

    gst_sample_unref(sample);

    copy_ms  += chrono::duration<double, std::milli>(
        chrono::high_resolution_clock::now() - cs).count();
    total_ms += chrono::duration<double, std::milli>(
        chrono::high_resolution_clock::now() - decode_start).count();

    return frame->valid;
}

//////////////////////////////////////////////////////////////////////////////
// Video Stream — N-branch tee pipeline
//////////////////////////////////////////////////////////////////////////////

imp_status_t imp_video_open(imp_video_stream_t** stream,
                            imp_video_source_t* source,
                            imp_context_t* ctx,
                            const imp_video_decode_opts_t* opts) {
    if (!stream || !source || !ctx) return IMP_ERROR_INVALID_ARGUMENT;

    auto start = chrono::high_resolution_clock::now();

    gst_init(nullptr, nullptr);

    auto* s = new imp_video_stream_s();
    s->timing = &ctx->timing;

    // Determine source resolution from source config
    int src_w = (source->width  > 0) ? source->width  : 1280;
    int src_h = (source->height > 0) ? source->height : 720;

    bool is_file   = (source->type == IMP_SOURCE_FILE);
    bool is_camera = (source->type == IMP_SOURCE_CAMERA);

    // Build resolved branch list.
    // If no branches specified, create a single branch at source resolution.
    struct resolved_branch {
        int width;
        int height;
        std::string name;
    };
    std::vector<resolved_branch> resolved;

    if (opts && opts->branches && opts->branch_count > 0) {
        for (uint32_t i = 0; i < opts->branch_count; i++) {
            resolved_branch rb;
            int bw = (int)opts->branches[i].width;
            int bh = (int)opts->branches[i].height;

            // Layered resolution: explicit > model dims > source res
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

            rb.name = opts->branches[i].name ? opts->branches[i].name
                                             : ("branch_" + std::to_string(i));
            resolved.push_back(rb);
        }
    } else {
        // Default: single branch at source resolution
        resolved.push_back({src_w, src_h, "default"});
    }

    // Determine if we need a tee (multiple branches or single branch != source res)
    bool needs_tee = (resolved.size() > 1);

    // Build GStreamer pipeline string
    std::string pipeline_str;

    // Queue params differ for file vs camera
    std::string queue_file = "queue max-size-buffers=0 max-size-time=0 max-size-bytes=0";
    std::string queue_cam  = "queue max-size-buffers=2 leaky=downstream";
    std::string appsink_file = "emit-signals=false sync=false max-buffers=0 drop=false";
    std::string appsink_cam  = "emit-signals=false sync=false max-buffers=2 drop=true";
    std::string queue_str   = is_file ? queue_file : queue_cam;
    std::string sink_props  = is_file ? appsink_file : appsink_cam;

    // Audit mode: force single-buffer, synchronous appsink (disable GStreamer
    // pre-buffering to isolate the effect of pipeline parallelism).
    if (is_file && opts && opts->sync_appsink) {
        queue_str  = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=no";
        sink_props = "emit-signals=false sync=false max-buffers=1 drop=false";
    }

    // Audit mode: arbitrary bounded queue depth (overrides sync_appsink).
    if (is_file && opts && opts->queue_depth > 0) {
        uint32_t d = opts->queue_depth;
        queue_str  = "queue max-size-buffers=" + std::to_string(d) +
                     " max-size-time=0 max-size-bytes=0 leaky=no";
        sink_props = "emit-signals=false sync=false max-buffers=" +
                     std::to_string(d) + " drop=false";
    }

    // Source + decode + NV12 convert
    if (is_file) {
        pipeline_str =
            "filesrc location=\"" + source->path + "\" ! "
            "decodebin ! "
            "d3d11upload ! "
            "d3d11convert ! "
            "video/x-raw(memory:D3D11Memory),format=NV12,"
            "width=" + std::to_string(src_w) + ",height=" + std::to_string(src_h);
    } else {
        pipeline_str =
            "mfvideosrc ! "
            "video/x-raw,width=" + std::to_string(src_w) + ",height=" + std::to_string(src_h) + " ! "
            "d3d11upload ! "
            "d3d11convert ! "
            "video/x-raw(memory:D3D11Memory),format=NV12,"
            "width=" + std::to_string(src_w) + ",height=" + std::to_string(src_h);
    }

    if (needs_tee) {
        pipeline_str += " ! tee name=t";

        for (size_t i = 0; i < resolved.size(); i++) {
            auto& rb = resolved[i];
            std::string sink_name = "branch_" + std::to_string(i);

            pipeline_str += " t. ! " + queue_str + " ! ";

            // If branch resolution differs from source, add d3d11scale
            if (rb.width != src_w || rb.height != src_h) {
                pipeline_str +=
                    "d3d11scale ! "
                    "video/x-raw(memory:D3D11Memory),format=NV12,"
                    "width=" + std::to_string(rb.width) + ",height=" + std::to_string(rb.height) + " ! ";
            }

            pipeline_str +=
                "d3d11download ! "
                "video/x-raw,format=NV12,"
                "width=" + std::to_string(rb.width) + ",height=" + std::to_string(rb.height) + " ! "
                "appsink name=" + sink_name + " " + sink_props;
        }
    } else {
        // Single branch — no tee needed
        auto& rb = resolved[0];

        if (rb.width != src_w || rb.height != src_h) {
            pipeline_str +=
                " ! d3d11scale ! "
                "video/x-raw(memory:D3D11Memory),format=NV12,"
                "width=" + std::to_string(rb.width) + ",height=" + std::to_string(rb.height);
        }

        pipeline_str +=
            " ! d3d11download ! "
            "video/x-raw,format=NV12,"
            "width=" + std::to_string(rb.width) + ",height=" + std::to_string(rb.height) + " ! "
            "appsink name=branch_0 " + sink_props;
    }

    std::cout << "Pipeline: " << pipeline_str << std::endl;

    GError* error = nullptr;
    s->pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error || !s->pipeline) {
        ctx->last_error = error ? error->message : "pipeline creation failed";
        if (error) g_error_free(error);
        delete s;
        return IMP_ERROR_DECODE_FAILED;
    }

    // Retrieve appsinks and populate branch info
    s->branches.resize(resolved.size());
    for (size_t i = 0; i < resolved.size(); i++) {
        std::string sink_name = "branch_" + std::to_string(i);
        s->branches[i].appsink = gst_bin_get_by_name(GST_BIN(s->pipeline), sink_name.c_str());
        if (!s->branches[i].appsink) {
            ctx->last_error = "failed to get appsink: " + sink_name;
            gst_object_unref(s->pipeline);
            delete s;
            return IMP_ERROR_DECODE_FAILED;
        }
        s->branches[i].width  = resolved[i].width;
        s->branches[i].height = resolved[i].height;
        s->branches[i].name   = resolved[i].name;
        s->branches[i].frame.allocate(resolved[i].width, resolved[i].height);
    }

    gst_element_set_state(s->pipeline, GST_STATE_PLAYING);

    GstStateChangeReturn ret = gst_element_get_state(
        s->pipeline, nullptr, nullptr, 5 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        ctx->last_error = "pipeline failed to start";
        for (auto& b : s->branches) {
            if (b.appsink) gst_object_unref(b.appsink);
        }
        gst_object_unref(s->pipeline);
        delete s;
        return IMP_ERROR_DECODE_FAILED;
    }

    // Get actual dimensions from first branch negotiated caps
    GstPad* pad = gst_element_get_static_pad(s->branches[0].appsink, "sink");
    if (pad) {
        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (caps) {
            GstStructure* st = gst_caps_get_structure(caps, 0);
            int actual_w = 0, actual_h = 0;
            gst_structure_get_int(st, "width", &actual_w);
            gst_structure_get_int(st, "height", &actual_h);
            if (actual_w > 0) s->branches[0].width = actual_w;
            if (actual_h > 0) s->branches[0].height = actual_h;
            gst_caps_unref(caps);
        }
        gst_object_unref(pad);
    }

    s->use_hw_decode = true;

    auto end = chrono::high_resolution_clock::now();
    ctx->timing.video_open_ms = chrono::duration<double, std::milli>(end - start).count();

    // Print branch info
    for (size_t i = 0; i < s->branches.size(); i++) {
        auto& b = s->branches[i];
        std::cout << "Branch " << i << " [" << b.name << "]: "
                  << b.width << "x" << b.height << " NV12" << std::endl;
    }
    if (s->branches.size() > 1) {
        std::cout << "GPU resize via d3d11scale (on D3D11)" << std::endl;
    }

    *stream = s;
    return IMP_OK;
}

#else  // Linux — delegate to gst_loader.cpp

imp_status_t imp_video_open(imp_video_stream_t** stream,
                            imp_video_source_t* source,
                            imp_context_t* ctx,
                            const imp_video_decode_opts_t* opts) {
    gst_loader_init();
    return linux_video_open(stream, source, ctx, opts);
}

#endif  // _WIN32

//////////////////////////////////////////////////////////////////////////////
// Video Read Frame — branch-aware
//////////////////////////////////////////////////////////////////////////////

imp_status_t imp_video_read_frame(imp_tensor_t** tensor,
                                  imp_video_stream_t* stream,
                                  uint32_t branch_index) {
#ifdef _WIN32

    auto& branch = stream->branches[branch_index];

    bool ok = read_nv12_from_appsink(
        branch.appsink, &branch.frame,
        branch.total_decode_ms,
        branch.total_decode_pull_ms,
        branch.total_decode_copy_ms);

    if (!ok) return IMP_ERROR_STREAM_END;

    // Fill tensor wrapper pointing to the branch's frame data
    branch.tensor_cache.y_data  = branch.frame.y_plane.data();
    branch.tensor_cache.uv_data = branch.frame.uv_plane.data();
    branch.tensor_cache.width   = branch.frame.width;
    branch.tensor_cache.height  = branch.frame.height;
    branch.tensor_cache.format  = IMP_FORMAT_NV12;
    branch.tensor_cache.valid   = true;
    branch.tensor_cache.device_type = IMP_DEVICE_CPU;

    if (tensor) *tensor = &branch.tensor_cache;
    return IMP_OK;

#else  // Linux
    return linux_video_read_frame(tensor, stream, branch_index);
#endif
}

imp_status_t imp_video_read_frame_by_name(imp_tensor_t** tensor,
                                          imp_video_stream_t* stream,
                                          const char* branch_name) {
    if (!stream || !branch_name) return IMP_ERROR_INVALID_ARGUMENT;

    for (uint32_t i = 0; i < (uint32_t)stream->branches.size(); i++) {
        if (stream->branches[i].name == branch_name) {
            return imp_video_read_frame(tensor, stream, i);
        }
    }
    return IMP_ERROR_INVALID_ARGUMENT; // branch name not found
}

imp_status_t imp_video_start_async(imp_video_stream_t* stream,
                                   imp_video_frame_callback_t callback,
                                   void* user_data) {
    (void)stream; (void)callback; (void)user_data;
    return IMP_ERROR_INTERNAL; // Not yet implemented
}

void imp_video_stop(imp_video_stream_t* stream) {
    (void)stream;
}

imp_status_t imp_video_get_info(imp_video_stream_t* stream,
                                uint32_t* width, uint32_t* height,
                                float* fps, int64_t* frame_count) {
    if (!stream || stream->branches.empty()) return IMP_ERROR_INVALID_ARGUMENT;
    // Return first branch dimensions as the "primary" stream info
    if (width)       *width  = (uint32_t)stream->branches[0].width;
    if (height)      *height = (uint32_t)stream->branches[0].height;
    if (fps)         *fps    = 30.0f; // TODO: detect from stream
    if (frame_count) *frame_count = -1; // unknown for live
    return IMP_OK;
}

void imp_video_close(imp_video_stream_t* stream) {
#ifdef _WIN32
    if (!stream) return;
    if (stream->pipeline) {
        gst_element_set_state(stream->pipeline, GST_STATE_NULL);
        for (auto& b : stream->branches) {
            if (b.appsink) gst_object_unref(b.appsink);
        }
        gst_object_unref(stream->pipeline);
    }
    delete stream;
#else
    linux_video_close(stream);
#endif
}

//////////////////////////////////////////////////////////////////////////////
// Video Encoder (Windows-only — Linux stub returns IMP_ERROR_INTERNAL)
//////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32

imp_status_t imp_video_encoder_create(imp_video_encoder_t** encoder,
                                      uint32_t width, uint32_t height,
                                      imp_context_t* ctx,
                                      const imp_video_encode_opts_t* opts) {
    if (!encoder || !ctx) return IMP_ERROR_INVALID_ARGUMENT;

    auto* e = new imp_video_encoder_s();
    e->width   = (int)width;
    e->height  = (int)height;
    e->fps_num = (opts && opts->framerate > 0) ? (int)opts->framerate : 30;
    e->fps_den = 1;
    e->timing  = &ctx->timing;

    std::string output_path = (opts && opts->output_path) ? opts->output_path : "output.mp4";

    std::cout << "=== Opening GPU encoder: " << output_path << " ===" << std::endl;

    std::string pipeline =
        "appsrc name=src format=time is-live=true do-timestamp=true ! "
        "video/x-raw,format=NV12,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(e->fps_num) + "/" + std::to_string(e->fps_den) + " ! "
        "mfh264enc ! h264parse ! mp4mux ! filesink location=\"" + output_path + "\"";

    GError* error = nullptr;
    e->pipeline = gst_parse_launch(pipeline.c_str(), &error);
    if (error) {
        ctx->last_error = error->message;
        g_error_free(error);
        delete e;
        return IMP_ERROR_ENCODE_FAILED;
    }

    e->appsrc = gst_bin_get_by_name(GST_BIN(e->pipeline), "src");
    if (!e->appsrc) {
        gst_object_unref(e->pipeline);
        delete e;
        return IMP_ERROR_ENCODE_FAILED;
    }

    g_object_set(e->appsrc, "stream-type", 0, "format", GST_FORMAT_TIME, nullptr);

    GstStateChangeReturn ret = gst_element_set_state(e->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        gst_object_unref(e->appsrc);
        gst_object_unref(e->pipeline);
        delete e;
        return IMP_ERROR_ENCODE_FAILED;
    }
    gst_element_get_state(e->pipeline, nullptr, nullptr, 500 * GST_MSECOND);

    e->encoder_name    = "mfh264enc";
    e->is_gpu_encoder  = true;
    e->initialized     = true;

    std::cout << "GPU encoder opened: mfh264enc" << std::endl;

    *encoder = e;
    return IMP_OK;
}

/**
 * Encode a frame from an imp_tensor_t (NV12 format).
 * The tensor must have valid y_data and uv_data pointers.
 */
imp_status_t imp_video_encoder_write(imp_video_encoder_t* encoder,
                                     imp_tensor_t* tensor) {
    if (!encoder || !encoder->initialized || !tensor || !tensor->valid)
        return IMP_ERROR_INVALID_ARGUMENT;

    auto encode_start = chrono::high_resolution_clock::now();

    int w = tensor->width;
    int h = tensor->height;
    size_t y_size  = (size_t)w * h;
    size_t uv_size = (size_t)w * (h / 2);

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, y_size + uv_size, nullptr);
    if (!buffer) return IMP_ERROR_ENCODE_FAILED;

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return IMP_ERROR_ENCODE_FAILED;
    }

    memcpy(map.data, tensor->y_data, y_size);
    memcpy(map.data + y_size, tensor->uv_data, uv_size);
    gst_buffer_unmap(buffer, &map);

    GstClockTime duration = gst_util_uint64_scale(GST_SECOND,
                                                  encoder->fps_den, encoder->fps_num);
    GST_BUFFER_PTS(buffer)      = encoder->frame_count * duration;
    GST_BUFFER_DURATION(buffer) = duration;

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(encoder->appsrc), buffer);
    if (ret != GST_FLOW_OK) return IMP_ERROR_ENCODE_FAILED;

    encoder->frame_count++;

    if (encoder->timing) {
        encoder->timing->total_encode_ms += chrono::duration<double, std::milli>(
            chrono::high_resolution_clock::now() - encode_start).count();
    }

    return IMP_OK;
}

void imp_video_encoder_close(imp_video_encoder_t* encoder) {
    if (!encoder || !encoder->initialized) return;

    gst_app_src_end_of_stream(GST_APP_SRC(encoder->appsrc));

    GstBus* bus = gst_element_get_bus(encoder->pipeline);
    if (bus) {
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
            (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (msg) gst_message_unref(msg);
        gst_object_unref(bus);
    }

    gst_element_set_state(encoder->pipeline, GST_STATE_NULL);
    gst_object_unref(encoder->appsrc);
    gst_object_unref(encoder->pipeline);

    std::cout << "Encoder closed. Frames: " << encoder->frame_count << std::endl;
    delete encoder;
}

#else  // Linux stubs for video encoder

imp_status_t imp_video_encoder_create(imp_video_encoder_t** encoder,
                                      uint32_t width, uint32_t height,
                                      imp_context_t* ctx,
                                      const imp_video_encode_opts_t* opts) {
    (void)encoder; (void)width; (void)height; (void)ctx; (void)opts;
    return IMP_ERROR_INTERNAL;  // Not yet implemented on Linux
}

imp_status_t imp_video_encoder_write(imp_video_encoder_t* encoder,
                                     imp_tensor_t* tensor) {
    (void)encoder; (void)tensor;
    return IMP_ERROR_INTERNAL;
}

void imp_video_encoder_close(imp_video_encoder_t* encoder) {
    (void)encoder;
}

#endif  // _WIN32

//////////////////////////////////////////////////////////////////////////////
// Tensor utilities
//////////////////////////////////////////////////////////////////////////////

imp_status_t imp_tensor_get_device_type(imp_tensor_t* tensor,
                                        imp_device_type_t* device_type) {
    if (!tensor || !device_type) return IMP_ERROR_INVALID_ARGUMENT;
    *device_type = tensor->device_type;
    return IMP_OK;
}

imp_status_t imp_tensor_get_device_name(imp_tensor_t* tensor,
                                        const char** device_name) {
    if (!tensor || !device_name) return IMP_ERROR_INVALID_ARGUMENT;
    *device_name = tensor->device_name.c_str();
    return IMP_OK;
}

imp_status_t imp_tensor_get_context_type(imp_tensor_t* tensor,
                                         imp_context_type_t* context_type) {
    (void)tensor; (void)context_type;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_tensor_get_ov(imp_tensor_t* tensor,
                               void** ov_tensor,
                               imp_device_type_t* device_type) {
    if (!tensor || !ov_tensor) return IMP_ERROR_INVALID_ARGUMENT;
    *ov_tensor = &tensor->ov_tensor;
    if (device_type) *device_type = tensor->device_type;
    return IMP_OK;
}

imp_status_t imp_tensor_get_shape(imp_tensor_t* tensor,
                                  int64_t* dims,
                                  size_t* num_dims) {
    if (!tensor || !dims || !num_dims) return IMP_ERROR_INVALID_ARGUMENT;

    // For NV12 tensors, report [height, width] as the shape
    if (tensor->format == IMP_FORMAT_NV12) {
        size_t n = std::min(*num_dims, (size_t)2);
        if (n >= 1) dims[0] = tensor->height;
        if (n >= 2) dims[1] = tensor->width;
        *num_dims = 2;
        return IMP_OK;
    }

    // For ov::Tensor backed tensors
    auto shape = tensor->ov_tensor.get_shape();
    size_t n = std::min(*num_dims, shape.size());
    for (size_t i = 0; i < n; i++) dims[i] = (int64_t)shape[i];
    *num_dims = shape.size();
    return IMP_OK;
}

imp_status_t imp_tensor_get_element_type(imp_tensor_t* tensor,
                                         imp_element_type_t* type) {
    if (!tensor || !type) return IMP_ERROR_INVALID_ARGUMENT;
    // NV12 tensors are always U8
    if (tensor->format == IMP_FORMAT_NV12) {
        *type = IMP_TYPE_U8;
        return IMP_OK;
    }
    return IMP_ERROR_INTERNAL; // TODO: map ov::element::Type
}

imp_status_t imp_tensor_get_nv12_planes(imp_tensor_t* tensor,
                                        const uint8_t** y_data,
                                        const uint8_t** uv_data,
                                        int* width,
                                        int* height) {
    if (!tensor || !y_data || !uv_data || !width || !height)
        return IMP_ERROR_INVALID_ARGUMENT;
    if (tensor->memory_type != IMP_MEM_SYSTEM || tensor->format != IMP_FORMAT_NV12)
        return IMP_ERROR_INVALID_ARGUMENT;
    *y_data  = tensor->y_data;
    *uv_data = tensor->uv_data;
    *width   = tensor->width;
    *height  = tensor->height;
    return IMP_OK;
}

imp_tensor_memory_type_t imp_tensor_get_memory_type(imp_tensor_t* tensor) {
    if (!tensor) return IMP_MEM_SYSTEM;
    return tensor->memory_type;
}

imp_status_t imp_tensor_get_va_surface(imp_tensor_t* tensor,
                                       uint32_t* surface_id,
                                       void**    va_display,
                                       int*      width,
                                       int*      height) {
    if (!tensor || !surface_id || !va_display || !width || !height)
        return IMP_ERROR_INVALID_ARGUMENT;
    if (tensor->memory_type != IMP_MEM_VA_SURFACE)
        return IMP_ERROR_INVALID_ARGUMENT;
    *surface_id = tensor->va_surface_id;
    *va_display = tensor->va_display;
    *width      = tensor->width;
    *height     = tensor->height;
    return IMP_OK;
}

void imp_tensor_release(imp_tensor_t* tensor) {
    if (!tensor) return;
    delete tensor;
}

void imp_free(void* ptr) {
    free(ptr);
}

//////////////////////////////////////////////////////////////////////////////
// HW support queries
//////////////////////////////////////////////////////////////////////////////

imp_status_t imp_hw_decode_supported(imp_context_t* ctx, bool* supported) {
    (void)ctx;
    if (supported) *supported = true;
    return IMP_OK;
}

imp_status_t imp_hw_encode_supported(imp_context_t* ctx, bool* supported) {
    (void)ctx;
    if (supported) *supported = true;
    return IMP_OK;
}

#ifndef _WIN32
bool imp_video_va_available(void) {
    gst_loader_init();  // no-op if already initialised
    return gst_loader_va_available();
}
void* imp_video_va_display(void) {
    gst_loader_init();
    return gst_loader_va_display();
}
void imp_video_set_va_display(void* va_display) {
    gst_loader_set_va_display(va_display);
}
#else
bool imp_video_va_available(void) { return false; }
void* imp_video_va_display(void)  { return nullptr; }
void imp_video_set_va_display(void* /*va_display*/) {}
#endif

// ============================================================================
// Image decode / encode and audio decode / encode — Windows-only.
// These functions use direct GStreamer calls that are only available on
// Windows (D3D11 pipeline, mfh264enc, etc.).  Linux stubs follow at the
// end of the #ifdef _WIN32 / #else block.
// ============================================================================
#ifdef _WIN32

//////////////////////////////////////////////////////////////////////////////
// Image decode / encode
//////////////////////////////////////////////////////////////////////////////

// Map imp_pixel_format_t → GStreamer video format string
static const char* imp_format_to_gst(imp_pixel_format_t fmt) {
    switch (fmt) {
        case IMP_FORMAT_RGB:  return "RGB";
        case IMP_FORMAT_BGR:  return "BGR";
        case IMP_FORMAT_RGBA: return "RGBA";
        case IMP_FORMAT_BGRA: return "BGRA";
        case IMP_FORMAT_NV12: return "NV12";
        case IMP_FORMAT_I420: return "I420";
        case IMP_FORMAT_GRAY: return "GRAY8";
        default:              return "RGB";
    }
}

// Channels for a given pixel format (planar formats return 0)
static int imp_format_channels(imp_pixel_format_t fmt) {
    switch (fmt) {
        case IMP_FORMAT_RGB:  return 3;
        case IMP_FORMAT_BGR:  return 3;
        case IMP_FORMAT_RGBA: return 4;
        case IMP_FORMAT_BGRA: return 4;
        case IMP_FORMAT_GRAY: return 1;
        default:              return 0;
    }
}

// Internal: pull decoded image sample from appsink, populate tensor
static imp_status_t imp_image_decode_pull(GstElement* pipeline,
                                          GstElement* appsink,
                                          imp_tensor_t** out_tensor,
                                          imp_pixel_format_t fmt,
                                          imp_layout_t layout,
                                          imp_element_type_t elem_type) {
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        return IMP_ERROR_DECODE_FAILED;
    }

    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 10 * GST_SECOND);
    if (!sample) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        return IMP_ERROR_DECODE_FAILED;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps*   caps   = gst_sample_get_caps(sample);

    GstVideoInfo info;
    gst_video_info_from_caps(&info, caps);

    int w = info.width;
    int h = info.height;

    auto* tensor = new imp_tensor_s();
    tensor->width  = w;
    tensor->height = h;
    tensor->format = fmt;
    tensor->device_type = IMP_DEVICE_CPU;
    tensor->valid  = false;

    GstVideoFrame vframe;
    if (!gst_video_frame_map(&vframe, &info, buffer, GST_MAP_READ)) {
        delete tensor;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        return IMP_ERROR_DECODE_FAILED;
    }

    if (fmt == IMP_FORMAT_NV12) {
        size_t y_size  = (size_t)w * h;
        size_t uv_size = (size_t)w * (h / 2);
        uint8_t* y_buf  = (uint8_t*)malloc(y_size);
        uint8_t* uv_buf = (uint8_t*)malloc(uv_size);
        if (!y_buf || !uv_buf) {
            free(y_buf); free(uv_buf);
            gst_video_frame_unmap(&vframe);
            delete tensor;
            gst_sample_unref(sample);
            gst_element_set_state(pipeline, GST_STATE_NULL);
            return IMP_ERROR_OUT_OF_MEMORY;
        }

        uint8_t* y_src  = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
        int y_stride    = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
        for (int row = 0; row < h; row++)
            memcpy(y_buf + row * w, y_src + row * y_stride, w);

        uint8_t* uv_src = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1);
        int uv_stride   = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);
        for (int row = 0; row < h / 2; row++)
            memcpy(uv_buf + row * w, uv_src + row * uv_stride, w);

        tensor->y_data  = y_buf;
        tensor->uv_data = uv_buf;
        tensor->valid   = true;
    } else {
        int channels = imp_format_channels(fmt);
        if (channels == 0) channels = 3;

        uint8_t* src_data = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
        int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);

        if (elem_type == IMP_TYPE_FP32) {
            size_t fp32_size = (size_t)w * h * channels;
            std::vector<float> fp32_data(fp32_size);
            for (int row = 0; row < h; row++) {
                uint8_t* row_ptr = src_data + row * stride;
                for (int col = 0; col < w * channels; col++)
                    fp32_data[row * w * channels + col] = row_ptr[col] / 255.0f;
            }

            if (layout == IMP_LAYOUT_NCHW) {
                ov::Shape shape = {1, (size_t)channels, (size_t)h, (size_t)w};
                tensor->ov_tensor = ov::Tensor(ov::element::f32, shape);
                float* dst = tensor->ov_tensor.data<float>();
                for (int c = 0; c < channels; c++)
                    for (int row = 0; row < h; row++)
                        for (int col = 0; col < w; col++)
                            dst[c * h * w + row * w + col] = fp32_data[row * w * channels + col + c];
            } else {
                ov::Shape shape = {1, (size_t)h, (size_t)w, (size_t)channels};
                tensor->ov_tensor = ov::Tensor(ov::element::f32, shape);
                memcpy(tensor->ov_tensor.data<float>(), fp32_data.data(), fp32_size * sizeof(float));
            }
        } else {
            if (layout == IMP_LAYOUT_NCHW) {
                ov::Shape shape = {1, (size_t)channels, (size_t)h, (size_t)w};
                tensor->ov_tensor = ov::Tensor(ov::element::u8, shape);
                uint8_t* dst = tensor->ov_tensor.data<uint8_t>();
                for (int c = 0; c < channels; c++)
                    for (int row = 0; row < h; row++) {
                        uint8_t* row_ptr = src_data + row * stride;
                        for (int col = 0; col < w; col++)
                            dst[c * h * w + row * w + col] = row_ptr[col * channels + c];
                    }
            } else {
                ov::Shape shape = {1, (size_t)h, (size_t)w, (size_t)channels};
                tensor->ov_tensor = ov::Tensor(ov::element::u8, shape);
                uint8_t* dst = tensor->ov_tensor.data<uint8_t>();
                int row_bytes = w * channels;
                for (int row = 0; row < h; row++)
                    memcpy(dst + row * row_bytes, src_data + row * stride, row_bytes);
            }
        }
        tensor->valid = true;
    }

    gst_video_frame_unmap(&vframe);
    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);

    *out_tensor = tensor;
    return IMP_OK;
}

imp_status_t imp_decode_image_file(imp_tensor_t** t, const char* f,
                                   imp_context_t* c, const imp_image_decode_opts_t* o,
                                   imp_decode_callback_t cb, void* ud) {
    (void)cb; (void)ud;

    if (!t || !f) return IMP_ERROR_INVALID_ARGUMENT;

    gst_init(nullptr, nullptr);

    imp_pixel_format_t fmt     = (o && o->output_format) ? o->output_format : IMP_FORMAT_RGB;
    imp_layout_t       layout  = (o) ? o->output_layout : IMP_LAYOUT_NHWC;
    imp_element_type_t etype   = (o) ? o->output_type   : IMP_TYPE_U8;
    uint32_t rw = (o) ? o->resize_width  : 0;
    uint32_t rh = (o) ? o->resize_height : 0;

    const char* gst_fmt = imp_format_to_gst(fmt);

    // Normalize path (backslashes break gst_parse_launch)
    std::string path(f);
    for (auto& ch : path) if (ch == '\\') ch = '/';

    std::string pipe = "filesrc location=\"" + path + "\" ! decodebin ! videoconvert ! "
                       "video/x-raw,format=" + gst_fmt;
    if (rw > 0 && rh > 0) {
        pipe += " ! videoscale ! video/x-raw,width=" + std::to_string(rw) +
                ",height=" + std::to_string(rh);
    }
    pipe += " ! appsink name=sink sync=false emit-signals=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipe.c_str(), &error);
    if (error) {
        if (c) c->last_error = error->message;
        g_error_free(error);
        if (pipeline) gst_object_unref(pipeline);
        return IMP_ERROR_DECODE_FAILED;
    }

    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsink) {
        gst_object_unref(pipeline);
        return IMP_ERROR_DECODE_FAILED;
    }

    imp_status_t status = imp_image_decode_pull(pipeline, appsink, t, fmt, layout, etype);

    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    return status;
}

imp_status_t imp_decode_image(imp_tensor_t** t, const void* d, size_t s,
                              imp_context_t* c, const imp_image_decode_opts_t* o,
                              imp_decode_callback_t cb, void* ud) {
    (void)cb; (void)ud;

    if (!t || !d || s == 0) return IMP_ERROR_INVALID_ARGUMENT;

    gst_init(nullptr, nullptr);

    imp_pixel_format_t fmt     = (o && o->output_format) ? o->output_format : IMP_FORMAT_RGB;
    imp_layout_t       layout  = (o) ? o->output_layout : IMP_LAYOUT_NHWC;
    imp_element_type_t etype   = (o) ? o->output_type   : IMP_TYPE_U8;
    uint32_t rw = (o) ? o->resize_width  : 0;
    uint32_t rh = (o) ? o->resize_height : 0;

    const char* gst_fmt = imp_format_to_gst(fmt);

    std::string pipe = "appsrc name=src ! decodebin ! videoconvert ! "
                       "video/x-raw,format=" + std::string(gst_fmt);
    if (rw > 0 && rh > 0) {
        pipe += " ! videoscale ! video/x-raw,width=" + std::to_string(rw) +
                ",height=" + std::to_string(rh);
    }
    pipe += " ! appsink name=sink sync=false emit-signals=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipe.c_str(), &error);
    if (error) {
        if (c) c->last_error = error->message;
        g_error_free(error);
        if (pipeline) gst_object_unref(pipeline);
        return IMP_ERROR_DECODE_FAILED;
    }

    GstElement* appsrc  = gst_bin_get_by_name(GST_BIN(pipeline), "src");
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsrc || !appsink) {
        if (appsrc) gst_object_unref(appsrc);
        if (appsink) gst_object_unref(appsink);
        gst_object_unref(pipeline);
        return IMP_ERROR_DECODE_FAILED;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    GstBuffer* buf = gst_buffer_new_allocate(nullptr, s, nullptr);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
        memcpy(map.data, d, s);
        gst_buffer_unmap(buf, &map);
    }
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf);
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 10 * GST_SECOND);
    imp_status_t status;
    if (!sample) {
        status = IMP_ERROR_DECODE_FAILED;
    } else {
        GstBuffer* sbuf = gst_sample_get_buffer(sample);
        GstCaps*   caps = gst_sample_get_caps(sample);
        GstVideoInfo info;
        gst_video_info_from_caps(&info, caps);

        int w = info.width;
        int h = info.height;
        auto* tensor = new imp_tensor_s();
        tensor->width  = w;
        tensor->height = h;
        tensor->format = fmt;
        tensor->device_type = IMP_DEVICE_CPU;
        tensor->valid  = false;

        GstVideoFrame vframe;
        if (gst_video_frame_map(&vframe, &info, sbuf, GST_MAP_READ)) {
            if (fmt == IMP_FORMAT_NV12) {
                size_t y_size  = (size_t)w * h;
                size_t uv_size = (size_t)w * (h / 2);
                uint8_t* y_buf  = (uint8_t*)malloc(y_size);
                uint8_t* uv_buf = (uint8_t*)malloc(uv_size);
                if (y_buf && uv_buf) {
                    uint8_t* y_src = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
                    int y_stride   = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
                    for (int row = 0; row < h; row++)
                        memcpy(y_buf + row * w, y_src + row * y_stride, w);
                    uint8_t* uv_src = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1);
                    int uv_stride   = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);
                    for (int row = 0; row < h / 2; row++)
                        memcpy(uv_buf + row * w, uv_src + row * uv_stride, w);
                    tensor->y_data  = y_buf;
                    tensor->uv_data = uv_buf;
                    tensor->valid   = true;
                } else {
                    free(y_buf); free(uv_buf);
                }
            } else {
                int channels = imp_format_channels(fmt);
                if (channels == 0) channels = 3;
                uint8_t* src_data = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
                int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);

                if (etype == IMP_TYPE_FP32) {
                    size_t fp32_count = (size_t)w * h * channels;
                    std::vector<float> fp32_data(fp32_count);
                    for (int row = 0; row < h; row++) {
                        uint8_t* rp = src_data + row * stride;
                        for (int col = 0; col < w * channels; col++)
                            fp32_data[row * w * channels + col] = rp[col] / 255.0f;
                    }
                    if (layout == IMP_LAYOUT_NCHW) {
                        ov::Shape shape = {1, (size_t)channels, (size_t)h, (size_t)w};
                        tensor->ov_tensor = ov::Tensor(ov::element::f32, shape);
                        float* dst = tensor->ov_tensor.data<float>();
                        for (int ch = 0; ch < channels; ch++)
                            for (int row = 0; row < h; row++)
                                for (int col = 0; col < w; col++)
                                    dst[ch * h * w + row * w + col] = fp32_data[row * w * channels + col + ch];
                    } else {
                        ov::Shape shape = {1, (size_t)h, (size_t)w, (size_t)channels};
                        tensor->ov_tensor = ov::Tensor(ov::element::f32, shape);
                        memcpy(tensor->ov_tensor.data<float>(), fp32_data.data(), fp32_count * sizeof(float));
                    }
                } else {
                    if (layout == IMP_LAYOUT_NCHW) {
                        ov::Shape shape = {1, (size_t)channels, (size_t)h, (size_t)w};
                        tensor->ov_tensor = ov::Tensor(ov::element::u8, shape);
                        uint8_t* dst = tensor->ov_tensor.data<uint8_t>();
                        for (int ch = 0; ch < channels; ch++)
                            for (int row = 0; row < h; row++) {
                                uint8_t* rp = src_data + row * stride;
                                for (int col = 0; col < w; col++)
                                    dst[ch * h * w + row * w + col] = rp[col * channels + ch];
                            }
                    } else {
                        ov::Shape shape = {1, (size_t)h, (size_t)w, (size_t)channels};
                        tensor->ov_tensor = ov::Tensor(ov::element::u8, shape);
                        uint8_t* dst = tensor->ov_tensor.data<uint8_t>();
                        int row_bytes = w * channels;
                        for (int row = 0; row < h; row++)
                            memcpy(dst + row * row_bytes, src_data + row * stride, row_bytes);
                    }
                }
                tensor->valid = true;
            }
            gst_video_frame_unmap(&vframe);
        }

        gst_sample_unref(sample);
        if (tensor->valid) {
            *t = tensor;
            status = IMP_OK;
        } else {
            delete tensor;
            status = IMP_ERROR_DECODE_FAILED;
        }
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    return status;
}

imp_status_t imp_decode_audio(imp_tensor_t** t, const void* d, size_t s,
                              imp_context_t* c, const imp_audio_decode_opts_t* o,
                              imp_decode_callback_t cb, void* ud) {
    (void)c; (void)cb; (void)ud;
    if (!t || !d || s == 0 || !o) return IMP_ERROR_INVALID_ARGUMENT;

    gst_init(nullptr, nullptr);

    uint32_t sampleRate = o->sample_rate > 0 ? o->sample_rate : 16000;
    uint32_t channels   = o->channels   > 0 ? o->channels   : 1;

    std::string pipeStr =
        "appsrc name=src ! decodebin ! audioconvert ! audioresample ! "
        "audio/x-raw,format=F32LE,channels=" + std::to_string(channels) +
        ",rate=" + std::to_string(sampleRate) +
        " ! appsink name=sink";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeStr.c_str(), &error);
    if (error || !pipeline) {
        if (error) g_error_free(error);
        if (pipeline) gst_object_unref(pipeline);
        return IMP_ERROR_DECODE_FAILED;
    }

    GstElement* appsrc  = gst_bin_get_by_name(GST_BIN(pipeline), "src");
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsrc || !appsink) {
        if (appsrc)  gst_object_unref(appsrc);
        if (appsink) gst_object_unref(appsink);
        gst_object_unref(pipeline);
        return IMP_ERROR_INTERNAL;
    }

    g_object_set(appsrc, "is-live", FALSE, nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Push input buffer
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, s, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, d, s);
    gst_buffer_unmap(buffer, &map);
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

    // Pull decoded samples
    std::vector<float> samples;
    for (;;) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 10 * GST_SECOND);
        if (!sample) {
            if (gst_app_sink_is_eos(GST_APP_SINK(appsink))) break;
            break;
        }
        GstBuffer* outBuf = gst_sample_get_buffer(sample);
        GstMapInfo outMap;
        if (gst_buffer_map(outBuf, &outMap, GST_MAP_READ)) {
            size_t nFloats = outMap.size / sizeof(float);
            const float* fptr = reinterpret_cast<const float*>(outMap.data);
            samples.insert(samples.end(), fptr, fptr + nFloats);
            gst_buffer_unmap(outBuf, &outMap);
        }
        gst_sample_unref(sample);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);

    if (samples.empty()) return IMP_ERROR_DECODE_FAILED;

    // Wrap in ov::Tensor and return as imp_tensor_t
    auto* tensor = new imp_tensor_t();
    tensor->ov_tensor = ov::Tensor(ov::element::f32, {1, samples.size()});
    memcpy(tensor->ov_tensor.data<float>(), samples.data(), samples.size() * sizeof(float));
    tensor->device_type = IMP_DEVICE_CPU;
    tensor->valid = true;
    *t = tensor;
    return IMP_OK;
}

imp_status_t imp_decode_audio_file(imp_tensor_t** t, const char* f,
                                   imp_context_t* c, const imp_audio_decode_opts_t* o,
                                   imp_decode_callback_t cb, void* ud) {
    (void)t;(void)f;(void)c;(void)o;(void)cb;(void)ud;
    return IMP_ERROR_INTERNAL; // TODO: implement - decode file to tensor with all samples
}

imp_status_t imp_audio_file_info(const char* file_path,
                                 uint32_t* sample_rate,
                                 uint32_t* channels,
                                 double* duration_sec) {
    if (!file_path) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    // Convert to URI
    std::string uri = "file:///";
    for (const char* p = file_path; *p; ++p) {
        if (*p == '\\') uri += '/';
        else if (*p == ' ') uri += "%20";
        else uri += *p;
    }
    
    GError* error = nullptr;
    GstDiscoverer* discoverer = gst_discoverer_new(10 * GST_SECOND, &error);
    if (!discoverer) {
        if (error) g_error_free(error);
        return IMP_ERROR_DECODE_FAILED;
    }
    
    GstDiscovererInfo* info = gst_discoverer_discover_uri(discoverer, uri.c_str(), &error);
    if (!info) {
        if (error) g_error_free(error);
        g_object_unref(discoverer);
        return IMP_ERROR_DECODE_FAILED;
    }
    
    // Get duration
    if (duration_sec) {
        GstClockTime dur = gst_discoverer_info_get_duration(info);
        *duration_sec = (double)dur / GST_SECOND;
    }
    
    // Get audio stream info
    GList* audio_streams = gst_discoverer_info_get_audio_streams(info);
    if (audio_streams) {
        GstDiscovererAudioInfo* audio_info = (GstDiscovererAudioInfo*)audio_streams->data;
        if (sample_rate) {
            *sample_rate = gst_discoverer_audio_info_get_sample_rate(audio_info);
        }
        if (channels) {
            *channels = gst_discoverer_audio_info_get_channels(audio_info);
        }
        gst_discoverer_stream_info_list_free(audio_streams);
    } else {
        // No audio stream found
        if (sample_rate) *sample_rate = 0;
        if (channels) *channels = 0;
    }
    
    gst_discoverer_info_unref(info);
    g_object_unref(discoverer);
    
    return IMP_OK;
}

//////////////////////////////////////////////////////////////////////////////
// Audio Stream Implementation (high-level decode → encode)
//////////////////////////////////////////////////////////////////////////////

// Helper: Get audio duration using GstDiscoverer
static double imp_audio_get_duration(const std::string& filepath) {
    GError* error = nullptr;
    
    std::string uri = "file:///";
    for (char c : filepath) {
        if (c == '\\') uri += '/';
        else if (c == ' ') uri += "%20";
        else uri += c;
    }
    
    GstDiscoverer* discoverer = gst_discoverer_new(10 * GST_SECOND, &error);
    if (!discoverer) {
        if (error) g_error_free(error);
        return 0.0;
    }
    
    GstDiscovererInfo* info = gst_discoverer_discover_uri(discoverer, uri.c_str(), &error);
    if (!info) {
        if (error) g_error_free(error);
        g_object_unref(discoverer);
        return 0.0;
    }
    
    GstClockTime duration = gst_discoverer_info_get_duration(info);
    double duration_sec = (double)duration / GST_SECOND;
    
    gst_discoverer_info_unref(info);
    g_object_unref(discoverer);
    
    return duration_sec;
}

// Forward declarations for helpers defined in the Audio Encoder section below
static std::string imp_audio_get_encoder(const std::string& codec);
static std::string imp_audio_get_muxer(const std::string& codec);

imp_status_t imp_audio_open(imp_audio_stream_t** stream,
                            const char* input_path,
                            const char* output_path,
                            const imp_audio_stream_opts_t* opts) {
    if (!stream || !input_path) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto* s = new imp_audio_stream_s();
    s->input_path = input_path;
    if (output_path) {
        s->output_path = output_path;
    }
    
    if (opts) {
        s->output_sample_rate = opts->sample_rate > 0 ? opts->sample_rate : 44100;
        s->output_channels = opts->channels > 0 ? opts->channels : 2;
        s->output_codec = opts->output_codec ? opts->output_codec : "mp3";
        s->output_bitrate_kbps = opts->output_bitrate_kbps > 0 ? opts->output_bitrate_kbps : 192;
    }
    
    // Get duration from input file
    s->duration_sec = imp_audio_get_duration(input_path);
    if (s->duration_sec <= 0) {
        s->last_error = "Failed to discover audio duration";
        delete s;
        return IMP_ERROR_DECODE_FAILED;
    }
    
    s->sample_rate = s->output_sample_rate;
    s->channels = s->output_channels;
    s->num_samples = (int64_t)(s->duration_sec * s->output_sample_rate);
    
    s->initialized = true;
    *stream = s;
    return IMP_OK;
}

imp_status_t imp_audio_get_info(imp_audio_stream_t* stream, imp_audio_info_t* info) {
    if (!stream || !info) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    info->sample_rate = stream->sample_rate;
    info->channels = stream->channels;
    info->duration_sec = stream->duration_sec;
    info->num_samples = stream->num_samples;
    
    return IMP_OK;
}

imp_status_t imp_audio_process(imp_audio_stream_t* stream) {
    if (!stream || !stream->initialized) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    if (stream->output_path.empty()) {
        stream->last_error = "No output path specified";
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    // Convert paths to forward slashes for GStreamer
    std::string input_path = stream->input_path;
    std::string output_path = stream->output_path;
    std::replace(input_path.begin(), input_path.end(), '\\', '/');
    std::replace(output_path.begin(), output_path.end(), '\\', '/');
    
    std::string encoder = imp_audio_get_encoder(stream->output_codec);
    std::string muxer = imp_audio_get_muxer(stream->output_codec);
    
    // Build pipeline: filesrc → decodebin → audioconvert → audioresample → caps → encoder → [muxer] → filesink
    std::string pipeline_str = 
        "filesrc location=\"" + input_path + "\" ! "
        "decodebin ! "
        "audioconvert ! "
        "audioresample ! "
        "audio/x-raw,format=S16LE,channels=" + std::to_string(stream->output_channels) + 
        ",rate=" + std::to_string(stream->output_sample_rate) + " ! ";
    
    if (encoder == "lamemp3enc") {
        pipeline_str += "lamemp3enc bitrate=" + std::to_string(stream->output_bitrate_kbps) + " ! ";
    } else if (encoder == "avenc_aac") {
        pipeline_str += "avenc_aac bitrate=" + std::to_string(stream->output_bitrate_kbps * 1000) + " ! ";
    } else if (encoder == "opusenc") {
        pipeline_str += "opusenc bitrate=" + std::to_string(stream->output_bitrate_kbps * 1000) + " ! ";
    } else {
        pipeline_str += encoder + " ! ";
    }
    
    if (!muxer.empty()) {
        pipeline_str += muxer + " ! ";
    }
    
    pipeline_str += "filesink location=\"" + output_path + "\"";
    
    GError* error = nullptr;
    stream->pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    
    if (error || !stream->pipeline) {
        stream->last_error = error ? error->message : "Pipeline creation failed";
        if (error) g_error_free(error);
        return IMP_ERROR_DECODE_FAILED;
    }
    
    auto start_time = chrono::high_resolution_clock::now();
    
    GstStateChangeReturn ret = gst_element_set_state(stream->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        stream->last_error = "Failed to start pipeline";
        gst_object_unref(stream->pipeline);
        stream->pipeline = nullptr;
        return IMP_ERROR_DECODE_FAILED;
    }
    
    // Wait for EOS or error
    GstBus* bus = gst_element_get_bus(stream->pipeline);
    GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
        (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    
    auto end_time = chrono::high_resolution_clock::now();
    stream->wall_time_sec = chrono::duration<double>(end_time - start_time).count();
    
    imp_status_t status = IMP_OK;
    if (msg) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            stream->last_error = err ? err->message : "Unknown error";
            if (err) g_error_free(err);
            if (debug) g_free(debug);
            status = IMP_ERROR_ENCODE_FAILED;
        }
        gst_message_unref(msg);
    }
    
    gst_object_unref(bus);
    gst_element_set_state(stream->pipeline, GST_STATE_NULL);
    gst_object_unref(stream->pipeline);
    stream->pipeline = nullptr;
    
    stream->processed = (status == IMP_OK);
    return status;
}

imp_status_t imp_audio_get_timing(imp_audio_stream_t* stream,
                                  double* wall_time_sec,
                                  double* realtime_factor) {
    if (!stream) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    if (wall_time_sec) {
        *wall_time_sec = stream->wall_time_sec;
    }
    if (realtime_factor) {
        if (stream->wall_time_sec > 0) {
            *realtime_factor = stream->duration_sec / stream->wall_time_sec;
        } else {
            *realtime_factor = 0.0;
        }
    }
    
    return IMP_OK;
}

void imp_audio_close(imp_audio_stream_t* stream) {
    if (!stream) return;
    
    if (stream->pipeline) {
        gst_element_set_state(stream->pipeline, GST_STATE_NULL);
        gst_object_unref(stream->pipeline);
    }
    
    delete stream;
}

// Internal: prepare raw pixel buffer from tensor for GStreamer encode.
// Uses GstVideoInfo to compute correct stride-aligned buffer size.
static GstBuffer* imp_image_prepare_encode_buffer(imp_tensor_t* t,
                                                   imp_pixel_format_t src_fmt,
                                                   int w, int h, int channels,
                                                   std::string& caps_str) {
    const char* gst_fmt = imp_format_to_gst(src_fmt);

    GstVideoFormat vfmt = gst_video_format_from_string(gst_fmt);
    GstVideoInfo vinfo;
    gst_video_info_set_format(&vinfo, vfmt, w, h);
    size_t data_size = vinfo.size;

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, data_size, nullptr);
    if (!buffer) return nullptr;

    GstVideoFrame vframe;
    if (!gst_video_frame_map(&vframe, &vinfo, buffer, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return nullptr;
    }

    if (src_fmt == IMP_FORMAT_NV12) {
        uint8_t* dst_y = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
        int dst_y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
        for (int row = 0; row < h; row++)
            memcpy(dst_y + row * dst_y_stride, t->y_data + row * w, w);
        uint8_t* dst_uv = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 1);
        int dst_uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 1);
        for (int row = 0; row < h / 2; row++)
            memcpy(dst_uv + row * dst_uv_stride, t->uv_data + row * w, w);
    } else if (t->ov_tensor) {
        uint8_t* dst = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vframe, 0);
        int dst_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, 0);
        int src_row_bytes = w * channels;

        auto shape = t->ov_tensor.get_shape();
        bool is_nchw = (shape.size() == 4 && shape[1] == (size_t)channels &&
                        shape[2] == (size_t)h && shape[3] == (size_t)w);
        bool is_fp32 = (t->ov_tensor.get_element_type() == ov::element::f32);

        if (is_nchw) {
            if (is_fp32) {
                float* src = t->ov_tensor.data<float>();
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        for (int ch = 0; ch < channels; ch++)
                            dst[row * dst_stride + col * channels + ch] =
                                (uint8_t)(std::min(std::max(src[ch * h * w + row * w + col] * 255.0f, 0.0f), 255.0f));
            } else {
                uint8_t* src = t->ov_tensor.data<uint8_t>();
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        for (int ch = 0; ch < channels; ch++)
                            dst[row * dst_stride + col * channels + ch] =
                                src[ch * h * w + row * w + col];
            }
        } else {
            if (is_fp32) {
                float* src = t->ov_tensor.data<float>();
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < src_row_bytes; col++)
                        dst[row * dst_stride + col] =
                            (uint8_t)(std::min(std::max(src[row * src_row_bytes + col] * 255.0f, 0.0f), 255.0f));
            } else {
                uint8_t* src = t->ov_tensor.data<uint8_t>();
                for (int row = 0; row < h; row++)
                    memcpy(dst + row * dst_stride, src + row * src_row_bytes, src_row_bytes);
            }
        }
    }

    gst_video_frame_unmap(&vframe);

    caps_str = "video/x-raw,format=" + std::string(gst_fmt) +
               ",width=" + std::to_string(w) +
               ",height=" + std::to_string(h) +
               ",framerate=1/1";

    return buffer;
}

imp_status_t imp_encode_image(void** d, size_t* s, imp_tensor_t* t,
                              imp_context_t* c, const imp_image_encode_opts_t* o,
                              imp_encode_callback_t cb, void* ud) {
    (void)cb; (void)ud;

    if (!d || !s || !t || !t->valid) return IMP_ERROR_INVALID_ARGUMENT;

    gst_init(nullptr, nullptr);

    std::string format_str = (o && o->format) ? o->format : "jpeg";
    std::string encoder_elem;
    if (format_str == "jpeg" || format_str == "jpg")  encoder_elem = "jpegenc";
    else if (format_str == "png")                      encoder_elem = "pngenc";
    else return IMP_ERROR_UNSUPPORTED_FORMAT;

    int quality = (o && o->quality > 0) ? o->quality : 90;

    int w = t->width, h = t->height;
    imp_pixel_format_t src_fmt = t->format;
    int channels = imp_format_channels(src_fmt);
    if (channels == 0 && src_fmt != IMP_FORMAT_NV12) channels = 3;

    if ((w == 0 || h == 0) && t->ov_tensor) {
        auto shape = t->ov_tensor.get_shape();
        if (shape.size() == 4) {
            if (shape[1] <= 4) { h = (int)shape[2]; w = (int)shape[3]; channels = (int)shape[1]; }
            else { h = (int)shape[1]; w = (int)shape[2]; channels = (int)shape[3]; }
        }
    }
    if (w == 0 || h == 0) return IMP_ERROR_INVALID_ARGUMENT;

    std::string caps_str;
    GstBuffer* buffer = imp_image_prepare_encode_buffer(t, src_fmt, w, h, channels, caps_str);
    if (!buffer) return IMP_ERROR_OUT_OF_MEMORY;

    GstElement* pipeline   = gst_pipeline_new("img-enc");
    GstElement* appsrc     = gst_element_factory_make("appsrc", "src");
    GstElement* convert    = gst_element_factory_make("videoconvert", "conv");
    GstElement* encoder    = gst_element_factory_make(encoder_elem.c_str(), "enc");
    GstElement* appsink    = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !appsrc || !convert || !encoder || !appsink) {
        gst_buffer_unref(buffer);
        if (pipeline) gst_object_unref(pipeline);
        return IMP_ERROR_ENCODE_FAILED;
    }

    GstCaps* src_caps = gst_caps_from_string(caps_str.c_str());
    g_object_set(appsrc, "caps", src_caps, "format", GST_FORMAT_TIME,
                 "stream-type", 0, "is-live", FALSE, nullptr);
    gst_caps_unref(src_caps);

    if (encoder_elem == "jpegenc")
        g_object_set(encoder, "quality", quality, nullptr);
    g_object_set(appsink, "sync", FALSE, "emit-signals", FALSE, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), appsrc, convert, encoder, appsink, nullptr);
    gst_element_link_many(appsrc, convert, encoder, appsink, nullptr);

    GST_BUFFER_PTS(buffer) = 0;
    GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 10 * GST_SECOND);
    imp_status_t status;
    if (!sample) {
        status = IMP_ERROR_ENCODE_FAILED;
    } else {
        GstBuffer* enc_buf = gst_sample_get_buffer(sample);
        GstMapInfo map;
        if (gst_buffer_map(enc_buf, &map, GST_MAP_READ)) {
            void* out = malloc(map.size);
            if (out) {
                memcpy(out, map.data, map.size);
                *d = out;
                *s = map.size;
                status = IMP_OK;
            } else {
                status = IMP_ERROR_OUT_OF_MEMORY;
            }
            gst_buffer_unmap(enc_buf, &map);
        } else {
            status = IMP_ERROR_ENCODE_FAILED;
        }
        gst_sample_unref(sample);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return status;
}

imp_status_t imp_encode_image_file(const char* f, imp_tensor_t* t,
                                   imp_context_t* c, const imp_image_encode_opts_t* o,
                                   imp_encode_callback_t cb, void* ud) {
    (void)cb; (void)ud;

    if (!f || !t || !t->valid) return IMP_ERROR_INVALID_ARGUMENT;

    gst_init(nullptr, nullptr);

    std::string format_str;
    if (o && o->format) {
        format_str = o->format;
    } else {
        std::string path(f);
        auto dot = path.rfind('.');
        if (dot != std::string::npos) {
            std::string ext = path.substr(dot + 1);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == "jpg" || ext == "jpeg") format_str = "jpeg";
            else if (ext == "png")             format_str = "png";
        }
    }
    if (format_str.empty()) format_str = "jpeg";

    std::string encoder_elem;
    if (format_str == "jpeg" || format_str == "jpg")  encoder_elem = "jpegenc";
    else if (format_str == "png")                      encoder_elem = "pngenc";
    else return IMP_ERROR_UNSUPPORTED_FORMAT;

    int quality = (o && o->quality > 0) ? o->quality : 90;

    int w = t->width, h = t->height;
    imp_pixel_format_t src_fmt = t->format;
    int channels = imp_format_channels(src_fmt);
    if (channels == 0 && src_fmt != IMP_FORMAT_NV12) channels = 3;

    if ((w == 0 || h == 0) && t->ov_tensor) {
        auto shape = t->ov_tensor.get_shape();
        if (shape.size() == 4) {
            if (shape[1] <= 4) { h = (int)shape[2]; w = (int)shape[3]; channels = (int)shape[1]; }
            else { h = (int)shape[1]; w = (int)shape[2]; channels = (int)shape[3]; }
        }
    }
    if (w == 0 || h == 0) return IMP_ERROR_INVALID_ARGUMENT;

    std::string caps_str;
    GstBuffer* buffer = imp_image_prepare_encode_buffer(t, src_fmt, w, h, channels, caps_str);
    if (!buffer) return IMP_ERROR_OUT_OF_MEMORY;

    std::string out_path(f);
    for (auto& ch : out_path) if (ch == '\\') ch = '/';

    GstElement* pipeline   = gst_pipeline_new("img-enc-file");
    GstElement* appsrc     = gst_element_factory_make("appsrc", "src");
    GstElement* convert    = gst_element_factory_make("videoconvert", "conv");
    GstElement* encoder    = gst_element_factory_make(encoder_elem.c_str(), "enc");
    GstElement* filesink   = gst_element_factory_make("filesink", "sink");

    if (!pipeline || !appsrc || !convert || !encoder || !filesink) {
        gst_buffer_unref(buffer);
        if (pipeline) gst_object_unref(pipeline);
        return IMP_ERROR_ENCODE_FAILED;
    }

    GstCaps* src_caps = gst_caps_from_string(caps_str.c_str());
    g_object_set(appsrc, "caps", src_caps, "format", GST_FORMAT_TIME,
                 "stream-type", 0, "is-live", FALSE, nullptr);
    gst_caps_unref(src_caps);

    if (encoder_elem == "jpegenc")
        g_object_set(encoder, "quality", quality, nullptr);
    g_object_set(filesink, "location", out_path.c_str(), nullptr);

    gst_bin_add_many(GST_BIN(pipeline), appsrc, convert, encoder, filesink, nullptr);
    gst_element_link_many(appsrc, convert, encoder, filesink, nullptr);

    GST_BUFFER_PTS(buffer) = 0;
    GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

    GstBus* bus = gst_element_get_bus(pipeline);
    imp_status_t status = IMP_ERROR_ENCODE_FAILED;
    if (bus) {
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, 10 * GST_SECOND,
            (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS)
                status = IMP_OK;
            gst_message_unref(msg);
        }
        gst_object_unref(bus);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return status;
}

//////////////////////////////////////////////////////////////////////////////
// Audio Encoder Implementation
//////////////////////////////////////////////////////////////////////////////

// Helper: Get encoder element for codec
static std::string imp_audio_get_encoder(const std::string& codec) {
    if (codec == "mp3")  return "lamemp3enc";
    if (codec == "aac")  return "avenc_aac";
    if (codec == "flac") return "flacenc";
    if (codec == "opus") return "opusenc";
    if (codec == "wav")  return "wavenc";
    return "lamemp3enc"; // default
}

// Helper: Get muxer element for codec (some don't need muxer)
static std::string imp_audio_get_muxer(const std::string& codec) {
    if (codec == "aac")  return "mp4mux";
    if (codec == "flac") return "";      // no muxer needed
    if (codec == "wav")  return "";      // wavenc is already containerized
    if (codec == "opus") return "oggmux";
    return ""; // mp3 with lamemp3enc doesn't need muxer
}

imp_status_t imp_audio_encoder_create(imp_audio_encoder_t** encoder,
                                      const imp_audio_encode_opts_t* opts,
                                      imp_encode_callback_t callback,
                                      void* user_data) {
    if (!encoder || !opts || !opts->output_path) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    auto* e = new imp_audio_encoder_s();
    e->output_path = opts->output_path;
    e->codec = opts->codec ? opts->codec : "mp3";
    e->bitrate_kbps = opts->bitrate_kbps > 0 ? opts->bitrate_kbps : 192;
    e->sample_rate = opts->sample_rate > 0 ? opts->sample_rate : 44100;
    e->channels = opts->channels > 0 ? opts->channels : 2;
    e->callback = callback;
    e->user_data = user_data;
    
    // Convert path to forward slashes for GStreamer
    std::string output_path = e->output_path;
    std::replace(output_path.begin(), output_path.end(), '\\', '/');
    
    // Get encoder element
    std::string enc_element = imp_audio_get_encoder(e->codec);
    std::string muxer = imp_audio_get_muxer(e->codec);
    
    // Build pipeline: appsrc → audioconvert → audioresample → encoder → [muxer] → filesink
    std::string pipeline_str = 
        "appsrc name=src format=time ! "
        "audioconvert ! "
        "audioresample ! "
        "audio/x-raw,format=S16LE,channels=" + std::to_string(e->channels) + 
        ",rate=" + std::to_string(e->sample_rate) + " ! ";
    
    // Add encoder with appropriate options
    if (enc_element == "lamemp3enc") {
        pipeline_str += "lamemp3enc bitrate=" + std::to_string(e->bitrate_kbps) + " ! ";
    } else if (enc_element == "avenc_aac") {
        pipeline_str += "avenc_aac bitrate=" + std::to_string(e->bitrate_kbps * 1000) + " ! ";
    } else if (enc_element == "opusenc") {
        pipeline_str += "opusenc bitrate=" + std::to_string(e->bitrate_kbps * 1000) + " ! ";
    } else {
        pipeline_str += enc_element + " ! ";
    }
    
    // Add muxer if needed
    if (!muxer.empty()) {
        pipeline_str += muxer + " ! ";
    }
    
    pipeline_str += "filesink location=\"" + output_path + "\"";
    
    // Create pipeline
    GError* error = nullptr;
    e->pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    
    if (error || !e->pipeline) {
        e->last_error = error ? error->message : "Pipeline creation failed";
        if (error) g_error_free(error);
        delete e;
        return IMP_ERROR_ENCODE_FAILED;
    }
    
    // Get appsrc element
    e->appsrc = gst_bin_get_by_name(GST_BIN(e->pipeline), "src");
    if (!e->appsrc) {
        e->last_error = "Failed to get appsrc element";
        gst_object_unref(e->pipeline);
        delete e;
        return IMP_ERROR_INTERNAL;
    }
    
    // Configure appsrc caps
    GstCaps* caps = gst_caps_new_simple("audio/x-raw",
        "format", G_TYPE_STRING, "S16LE",
        "rate", G_TYPE_INT, (int)e->sample_rate,
        "channels", G_TYPE_INT, (int)e->channels,
        "layout", G_TYPE_STRING, "interleaved",
        nullptr);
    g_object_set(e->appsrc, "caps", caps, "is-live", FALSE, nullptr);
    gst_caps_unref(caps);
    
    // Start pipeline
    GstStateChangeReturn ret = gst_element_set_state(e->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        e->last_error = "Failed to start pipeline";
        gst_object_unref(e->appsrc);
        gst_object_unref(e->pipeline);
        delete e;
        return IMP_ERROR_ENCODE_FAILED;
    }
    
    e->initialized = true;
    *encoder = e;
    return IMP_OK;
}

imp_status_t imp_audio_encoder_write(imp_audio_encoder_t* encoder,
                                     imp_tensor_t* tensor) {
    if (!encoder || !encoder->initialized || !tensor) {
        return IMP_ERROR_INVALID_ARGUMENT;
    }
    
    // TODO: Extract audio data from tensor and push to appsrc
    // For now, this is a stub - real implementation needs tensor audio data access
    (void)tensor;
    
    return IMP_ERROR_INTERNAL; // Not fully implemented yet
}

void imp_audio_encoder_close(imp_audio_encoder_t* encoder) {
    if (!encoder) return;
    
    if (encoder->appsrc) {
        // Signal end of stream
        gst_app_src_end_of_stream(GST_APP_SRC(encoder->appsrc));
    }
    
    if (encoder->pipeline) {
        // Wait for EOS to propagate
        GstBus* bus = gst_element_get_bus(encoder->pipeline);
        if (bus) {
            GstMessage* msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
                (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
            
            if (msg) {
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR && encoder->callback) {
                    // Call callback with error
                    encoder->callback(IMP_ERROR_ENCODE_FAILED, nullptr, 0, encoder->user_data);
                } else if (encoder->callback) {
                    // Call callback with success (no data returned for file output)
                    encoder->callback(IMP_OK, nullptr, 0, encoder->user_data);
                }
                gst_message_unref(msg);
            }
            gst_object_unref(bus);
        }
        
        gst_element_set_state(encoder->pipeline, GST_STATE_NULL);
        
        if (encoder->appsrc) {
            gst_object_unref(encoder->appsrc);
        }
        gst_object_unref(encoder->pipeline);
    }
    
    delete encoder;
}

// ---------------------------------------------------------------------------
// One-shot audio encode (to memory)
// ---------------------------------------------------------------------------

static imp_status_t imp_encode_audio_wav(void** data, size_t* data_size,
                                         const float* samples, size_t num_samples,
                                         uint32_t sampleRate, uint32_t channels) {
    uint32_t byteRate    = sampleRate * channels * sizeof(float);
    uint32_t blockAlign  = channels * static_cast<uint32_t>(sizeof(float));
    uint32_t dataSize    = static_cast<uint32_t>(num_samples * sizeof(float));
    uint32_t chunkSize   = 36 + dataSize;
    size_t   totalSize   = 44 + dataSize;

    uint8_t* buf = static_cast<uint8_t*>(malloc(totalSize));
    if (!buf) return IMP_ERROR_INTERNAL;

    auto w16 = [](uint8_t* p, uint16_t v) { p[0]=v&0xFF; p[1]=(v>>8)&0xFF; };
    auto w32 = [](uint8_t* p, uint32_t v) { p[0]=v&0xFF; p[1]=(v>>8)&0xFF; p[2]=(v>>16)&0xFF; p[3]=(v>>24)&0xFF; };

    memcpy(buf,      "RIFF", 4);  w32(buf+4, chunkSize);
    memcpy(buf+8,    "WAVE", 4);
    memcpy(buf+12,   "fmt ", 4);  w32(buf+16, 16);
    w16(buf+20, 3);  // IEEE float
    w16(buf+22, static_cast<uint16_t>(channels));
    w32(buf+24, sampleRate);
    w32(buf+28, byteRate);
    w16(buf+32, static_cast<uint16_t>(blockAlign));
    w16(buf+34, 32);  // bits per sample
    memcpy(buf+36, "data", 4);  w32(buf+40, dataSize);
    memcpy(buf+44, samples, dataSize);

    *data = buf;
    *data_size = totalSize;
    return IMP_OK;
}

imp_status_t imp_encode_audio(void** data,
                              size_t* data_size,
                              const float* samples,
                              size_t num_samples,
                              const imp_audio_encode_opts_t* opts) {
    if (!data || !data_size || !samples || num_samples == 0 || !opts)
        return IMP_ERROR_INVALID_ARGUMENT;

    std::string codec = opts->codec ? opts->codec : "wav";
    uint32_t sampleRate = opts->sample_rate > 0 ? opts->sample_rate : 16000;
    uint32_t channels   = opts->channels   > 0 ? opts->channels   : 1;

    if (codec == "wav") {
        return imp_encode_audio_wav(data, data_size, samples, num_samples, sampleRate, channels);
    }

    if (codec == "pcm") {
        size_t sz = num_samples * sizeof(float);
        *data = malloc(sz);
        if (!*data) return IMP_ERROR_INTERNAL;
        memcpy(*data, samples, sz);
        *data_size = sz;
        return IMP_OK;
    }

    // Lossy / lossless codecs via GStreamer
    gst_init(nullptr, nullptr);

    std::string encElement;
    std::string muxElement;
    uint32_t bitrate = opts->bitrate_kbps > 0 ? opts->bitrate_kbps : 192;

    if (codec == "mp3") {
        encElement = "lamemp3enc bitrate=" + std::to_string(bitrate);
    } else if (codec == "flac") {
        encElement = "flacenc";
    } else if (codec == "opus") {
        uint32_t opusRate = (sampleRate <= 8000) ? 8000 :
                            (sampleRate <= 12000) ? 12000 :
                            (sampleRate <= 16000) ? 16000 :
                            (sampleRate <= 24000) ? 24000 : 48000;
        encElement = "opusenc bitrate=" + std::to_string(bitrate * 1000);
        muxElement = "oggmux";
        sampleRate = opusRate;
    } else if (codec == "aac") {
        encElement = "avenc_aac bitrate=" + std::to_string(bitrate * 1000);
        muxElement = "aacparse ! adtsmux";
    } else {
        return IMP_ERROR_INVALID_ARGUMENT;
    }

    std::string pipeStr =
        "appsrc name=src format=time ! "
        "audioconvert ! audioresample ! "
        "audio/x-raw,format=S16LE,channels=" + std::to_string(channels) +
        ",rate=" + std::to_string(sampleRate) + " ! " +
        encElement + " ! ";
    if (!muxElement.empty()) pipeStr += muxElement + " ! ";
    pipeStr += "appsink name=sink";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeStr.c_str(), &error);
    if (error || !pipeline) {
        if (error) g_error_free(error);
        if (pipeline) gst_object_unref(pipeline);
        return IMP_ERROR_ENCODE_FAILED;
    }

    GstElement* appsrc  = gst_bin_get_by_name(GST_BIN(pipeline), "src");
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsrc || !appsink) {
        if (appsrc)  gst_object_unref(appsrc);
        if (appsink) gst_object_unref(appsink);
        gst_object_unref(pipeline);
        return IMP_ERROR_INTERNAL;
    }

    GstCaps* caps = gst_caps_new_simple("audio/x-raw",
        "format",  G_TYPE_STRING, "F32LE",
        "rate",    G_TYPE_INT, (int)(opts->sample_rate > 0 ? opts->sample_rate : 16000),
        "channels", G_TYPE_INT, (int)channels,
        "layout",  G_TYPE_STRING, "interleaved",
        nullptr);
    g_object_set(appsrc, "caps", caps, "is-live", FALSE, nullptr);
    gst_caps_unref(caps);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    size_t byteSize = num_samples * sizeof(float);
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, byteSize, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, samples, byteSize);
    gst_buffer_unmap(buffer, &map);
    GST_BUFFER_PTS(buffer)      = 0;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(
        num_samples / channels, GST_SECOND, opts->sample_rate > 0 ? opts->sample_rate : 16000);
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));

    std::vector<uint8_t> encoded;
    for (;;) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 10 * GST_SECOND);
        if (!sample) {
            if (gst_app_sink_is_eos(GST_APP_SINK(appsink))) break;
            break;
        }
        GstBuffer* outBuf = gst_sample_get_buffer(sample);
        GstMapInfo outMap;
        if (gst_buffer_map(outBuf, &outMap, GST_MAP_READ)) {
            encoded.insert(encoded.end(), outMap.data, outMap.data + outMap.size);
            gst_buffer_unmap(outBuf, &outMap);
        }
        gst_sample_unref(sample);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);

    if (encoded.empty()) return IMP_ERROR_ENCODE_FAILED;

    *data = malloc(encoded.size());
    if (!*data) return IMP_ERROR_INTERNAL;
    memcpy(*data, encoded.data(), encoded.size());
    *data_size = encoded.size();
    return IMP_OK;
}

imp_status_t imp_encode_audio_file(const char* file_path,
                                   const float* samples,
                                   size_t num_samples,
                                   const imp_audio_encode_opts_t* opts) {
    void* data = nullptr;
    size_t data_size = 0;
    imp_status_t st = imp_encode_audio(&data, &data_size, samples, num_samples, opts);
    if (st != IMP_OK) return st;

    FILE* fp = fopen(file_path, "wb");
    if (!fp) { free(data); return IMP_ERROR_INTERNAL; }
    fwrite(data, 1, data_size, fp);
    fclose(fp);
    free(data);
    return IMP_OK;
}

#else  // Linux stubs for image/audio decode-encode and audio stream/encoder

imp_status_t imp_decode_image_file(imp_tensor_t** t, const char* f,
                                   imp_context_t* c,
                                   const imp_image_decode_opts_t* o,
                                   imp_decode_callback_t cb, void* ud) {
    (void)t; (void)f; (void)c; (void)o; (void)cb; (void)ud;
    return IMP_ERROR_INTERNAL;  // Not yet implemented on Linux
}

imp_status_t imp_decode_image(imp_tensor_t** t, const void* d, size_t s,
                              imp_context_t* c,
                              const imp_image_decode_opts_t* o,
                              imp_decode_callback_t cb, void* ud) {
    (void)t; (void)d; (void)s; (void)c; (void)o; (void)cb; (void)ud;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_decode_audio(imp_tensor_t** t, const void* d, size_t s,
                              imp_context_t* c,
                              const imp_audio_decode_opts_t* o,
                              imp_decode_callback_t cb, void* ud) {
    (void)t; (void)d; (void)s; (void)c; (void)o; (void)cb; (void)ud;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_decode_audio_file(imp_tensor_t** t, const char* f,
                                   imp_context_t* c,
                                   const imp_audio_decode_opts_t* o,
                                   imp_decode_callback_t cb, void* ud) {
    (void)t; (void)f; (void)c; (void)o; (void)cb; (void)ud;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_file_info(const char* file_path,
                                 uint32_t* sample_rate,
                                 uint32_t* channels,
                                 double* duration_sec) {
    (void)file_path; (void)sample_rate; (void)channels; (void)duration_sec;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_open(imp_audio_stream_t** stream,
                            const char* input_path,
                            const char* output_path,
                            const imp_audio_stream_opts_t* opts) {
    (void)stream; (void)input_path; (void)output_path; (void)opts;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_get_info(imp_audio_stream_t* stream,
                                imp_audio_info_t* info) {
    (void)stream; (void)info;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_process(imp_audio_stream_t* stream) {
    (void)stream;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_get_timing(imp_audio_stream_t* stream,
                                  double* wall_time_sec,
                                  double* realtime_factor) {
    (void)stream; (void)wall_time_sec; (void)realtime_factor;
    return IMP_ERROR_INTERNAL;
}

void imp_audio_close(imp_audio_stream_t* stream) {
    (void)stream;
}

imp_status_t imp_encode_image(void** d, size_t* s, imp_tensor_t* t,
                              imp_context_t* c,
                              const imp_image_encode_opts_t* o,
                              imp_encode_callback_t cb, void* ud) {
    (void)d; (void)s; (void)t; (void)c; (void)o; (void)cb; (void)ud;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_encode_image_file(const char* f, imp_tensor_t* t,
                                   imp_context_t* c,
                                   const imp_image_encode_opts_t* o,
                                   imp_encode_callback_t cb, void* ud) {
    (void)f; (void)t; (void)c; (void)o; (void)cb; (void)ud;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_encoder_create(imp_audio_encoder_t** encoder,
                                      const imp_audio_encode_opts_t* opts,
                                      imp_encode_callback_t callback,
                                      void* user_data) {
    (void)encoder; (void)opts; (void)callback; (void)user_data;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_audio_encoder_write(imp_audio_encoder_t* encoder,
                                     imp_tensor_t* tensor) {
    (void)encoder; (void)tensor;
    return IMP_ERROR_INTERNAL;
}

void imp_audio_encoder_close(imp_audio_encoder_t* encoder) {
    (void)encoder;
}

imp_status_t imp_encode_audio(void** data,
                              size_t* data_size,
                              const float* samples,
                              size_t num_samples,
                              const imp_audio_encode_opts_t* opts) {
    (void)data; (void)data_size; (void)samples; (void)num_samples; (void)opts;
    return IMP_ERROR_INTERNAL;
}

imp_status_t imp_encode_audio_file(const char* file_path,
                                   const float* samples,
                                   size_t num_samples,
                                   const imp_audio_encode_opts_t* opts) {
    (void)file_path; (void)samples; (void)num_samples; (void)opts;
    return IMP_ERROR_INTERNAL;
}

#endif  // _WIN32
