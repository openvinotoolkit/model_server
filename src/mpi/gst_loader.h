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
 * GStreamer runtime loader for Linux.
 *
 * Provides a dlopen-based loading of GStreamer shared libraries so that
 * the Intel MPI binary can start even on systems without GStreamer installed.
 * All GStreamer operations are only available when gst_loader_available()
 * returns true.
 *
 * Usage:
 *   gst_loader_init();          // call once; safe to call multiple times
 *   if (!gst_loader_available()) return IMP_ERROR_DEVICE_NOT_AVAILABLE;
 *   // use linux_video_open() / linux_video_read_frame() / linux_video_close()
 *
 * This header intentionally does NOT include any GStreamer headers so that
 * files outside of gst_loader.cpp do not transitively link against GStreamer.
 * All GStreamer types exposed in the public API are forward-declared below.
 */

#ifndef GST_LOADER_H
#define GST_LOADER_H

#ifndef _WIN32

#include <cstdint>
#include <cstddef>

// Forward-declare GStreamer opaque types.  Consumers only hold pointers so
// full definitions are not required here.
typedef struct _GstElement GstElement;

// Intel MPI public API — provides all imp_* types and status codes.
#include "intel_mpi.h"

/**
 * Attempt to dlopen GStreamer shared libraries and resolve all required
 * function pointers.  Safe to call multiple times (idempotent after first
 * call).
 *
 * @return true  if all symbols were resolved and GStreamer is usable.
 * @return false if any library or symbol was missing.
 */
bool gst_loader_init();

/**
 * Returns true after a successful gst_loader_init().
 * Returns false if GStreamer is not installed or init has not been called.
 */
bool gst_loader_available();

/**
 * Returns true if VA-API (vapostproc) is available for GPU-accelerated decode.
 * Only valid after gst_loader_init() returns true.
 */
bool gst_loader_va_available();

/**
 * Override the process-wide VADisplay with one obtained from OV's GPU context.
 * Must be called before linux_video_open() when VA surface sharing is wanted.
 */
void gst_loader_set_va_display(void* va_display);

/**
 * Returns the process-wide VADisplay (cast to void*) created during init.
 * Only valid when VA surface sharing is enabled (libgstva + libva loaded).
 * Returns nullptr when VA surface sharing is not available.
 * Use with ov::intel_gpu::ocl::VAContext for zero-copy GPU inference.
 */
void* gst_loader_va_display();

// ---------------------------------------------------------------------------
// High-level Linux video operations.
// These are the only entry-points needed by intel_mpi.cpp on Linux.
// All GStreamer details are hidden inside gst_loader.cpp.
// ---------------------------------------------------------------------------

/**
 * Open a video file for decode.
 * Equivalent to imp_video_open() on Linux.
 *
 * @param stream   output: newly allocated stream handle
 * @param source   video source configuration (path, dimensions, …)
 * @param ctx      Intel MPI context (for dimensions hints and error reporting)
 * @param opts     decode options (branches, sync mode, …) — may be nullptr
 * @return IMP_OK on success, error code otherwise.
 */
imp_status_t linux_video_open(imp_video_stream_t** stream,
                              imp_video_source_t* source,
                              imp_context_t* ctx,
                              const imp_video_decode_opts_t* opts);

/**
 * Read one NV12 frame from the given branch of an open stream.
 * Equivalent to imp_video_read_frame() on Linux.
 *
 * @param tensor        output: pointer to internal tensor cache (not owned)
 * @param stream        open stream
 * @param branch_index  branch to read from (0 for single-branch streams)
 * @return IMP_OK, IMP_ERROR_STREAM_END when exhausted, or error code.
 */
imp_status_t linux_video_read_frame(imp_tensor_t** tensor,
                                    imp_video_stream_t* stream,
                                    uint32_t branch_index);

/**
 * Close and free a stream opened with linux_video_open().
 * Equivalent to imp_video_close() on Linux.
 */
void linux_video_close(imp_video_stream_t* stream);

#endif  // !_WIN32
#endif  // GST_LOADER_H
