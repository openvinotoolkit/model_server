/**
 * Intel Media Processing Interface (Intel MPI)
 * 
 * Lightweight API for media decode/encode with zero-copy OpenVINO integration.
 * Designed for Intel GPU acceleration with minimal CPU-GPU memory transfers.
 * 
 * Version: 0.1.0
 */

#ifndef INTEL_MPI_H
#define INTEL_MPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

//////////////////////////////////////////////////////////////////////////////
// Forward Declarations (OpenVINO C API types)
//////////////////////////////////////////////////////////////////////////////

typedef struct ov_compiled_model ov_compiled_model_t;
typedef struct ov_remote_context ov_remote_context_t;

//////////////////////////////////////////////////////////////////////////////
// Device Types
//////////////////////////////////////////////////////////////////////////////

typedef enum {
    IMP_DEVICE_AUTO = 0,    // Auto-detect from context (match inference device)
    IMP_DEVICE_CPU = 1,     // CPU (host memory)
    IMP_DEVICE_GPU = 2,     // Intel GPU
    IMP_DEVICE_NPU = 3,     // Intel NPU
} imp_device_type_t;

//////////////////////////////////////////////////////////////////////////////
// Tensor Handle
//////////////////////////////////////////////////////////////////////////////

/**
 * Opaque tensor handle (pointer type)
 * 
 * Wraps either ov::Tensor (CPU) or ov::RemoteTensor (GPU/NPU).
 * Use imp_tensor_get_*() functions to query properties.
 * 
 * NOTE: Despite the abstraction, the underlying tensor type matters:
 * - CPU context: wraps ov::Tensor (host memory)
 * - GPU/NPU context: wraps ov::RemoteTensor (device memory)
 * Use imp_tensor_get_device_type() to determine the actual type,
 * and imp_tensor_get_ov() to access the underlying OpenVINO tensor.
 */
typedef struct imp_tensor_s imp_tensor_t;

//////////////////////////////////////////////////////////////////////////////
// Version
//////////////////////////////////////////////////////////////////////////////

#define IMP_VERSION_MAJOR 0
#define IMP_VERSION_MINOR 1
#define IMP_VERSION_PATCH 0

//////////////////////////////////////////////////////////////////////////////
// Error Codes
//////////////////////////////////////////////////////////////////////////////

typedef enum {
    IMP_OK = 0,
    IMP_ERROR_INVALID_ARGUMENT = -1,
    IMP_ERROR_OUT_OF_MEMORY = -2,
    IMP_ERROR_DEVICE_NOT_AVAILABLE = -3,
    IMP_ERROR_UNSUPPORTED_FORMAT = -4,
    IMP_ERROR_DECODE_FAILED = -5,
    IMP_ERROR_ENCODE_FAILED = -6,
    IMP_ERROR_CONTEXT_MISMATCH = -7,
    IMP_ERROR_STREAM_END = -8,
    IMP_ERROR_TIMEOUT = -9,
    IMP_ERROR_INTERNAL = -99
} imp_status_t;

//////////////////////////////////////////////////////////////////////////////
// Context Types
//////////////////////////////////////////////////////////////////////////////

typedef enum {
    IMP_CONTEXT_OPENCL = 0,      // OpenCL context (Linux/Windows)
    IMP_CONTEXT_D3D11 = 1,       // Direct3D 11 (Windows)
    IMP_CONTEXT_VAAPI = 2,       // VA-API (Linux)
} imp_context_type_t;

//////////////////////////////////////////////////////////////////////////////
// Data Types
//////////////////////////////////////////////////////////////////////////////

typedef enum {
    IMP_TYPE_U8 = 0,
    IMP_TYPE_U16 = 1,
    IMP_TYPE_FP16 = 2,
    IMP_TYPE_FP32 = 3,
    IMP_TYPE_I8 = 4,
    IMP_TYPE_I32 = 5
} imp_element_type_t;

typedef enum {
    IMP_FORMAT_RGB = 0,
    IMP_FORMAT_BGR = 1,
    IMP_FORMAT_NV12 = 2,        // YUV 4:2:0, two planes (Y + interleaved UV)
    IMP_FORMAT_I420 = 3,        // YUV 4:2:0, three planes
    IMP_FORMAT_GRAY = 4,
    IMP_FORMAT_RGBA = 5,
    IMP_FORMAT_BGRA = 6
} imp_pixel_format_t;

typedef enum {
    IMP_LAYOUT_NHWC = 0,        // [batch, height, width, channels]
    IMP_LAYOUT_NCHW = 1,        // [batch, channels, height, width]
    IMP_LAYOUT_HWC = 2,         // [height, width, channels]
    IMP_LAYOUT_CHW = 3          // [channels, height, width]
} imp_layout_t;

//////////////////////////////////////////////////////////////////////////////
// Context
//////////////////////////////////////////////////////////////////////////////

/**
 * Opaque context handle (pointer type)
 */
typedef struct imp_context_s imp_context_t;

/**
 * Create context from OpenVINO compiled model
 * Extracts GPU context from model for zero-copy tensor sharing.
 * 
 * @param ctx Output: context handle pointer
 * @param compiled_model OpenVINO compiled model (must be GPU-compiled)
 * @return Status code
 */
imp_status_t imp_context_create(imp_context_t** ctx, 
                                ov_compiled_model_t* compiled_model);

/**
 * Create context from OpenVINO remote context
 * Use when sharing context between multiple models.
 * 
 * @param ctx Output: context handle pointer
 * @param remote_ctx OpenVINO remote context
 * @param type Context backend type
 * @return Status code
 */
imp_status_t imp_context_create_from_remote(imp_context_t** ctx,
                                            ov_remote_context_t* remote_ctx,
                                            imp_context_type_t type);

/**
 * Get underlying native handle
 * 
 * @param ctx Context handle
 * @param type Output: context type
 * @param native_handle Output: native handle (cl_context, ID3D11Device*, etc.)
 * @return Status code
 */
imp_status_t imp_context_get_native(imp_context_t* ctx,
                                    imp_context_type_t* type,
                                    void** native_handle);

/**
 * Get OpenVINO remote context
 * 
 * @param ctx Context handle
 * @param remote_ctx Output: OpenVINO remote context
 * @return Status code
 */
imp_status_t imp_context_get_ov_remote(imp_context_t* ctx,
                                       ov_remote_context_t** remote_ctx);

/**
 * Get device type the context is configured for
 * 
 * @param ctx Context handle
 * @param device_type Output: device type
 * @return Status code
 */
imp_status_t imp_context_get_device_type(imp_context_t* ctx,
                                          imp_device_type_t* device_type);

/**
 * Get device name (e.g., "GPU.0", "GPU.1", "CPU", "NPU")
 * 
 * @param ctx Context handle
 * @param device_name Output: device name string (valid until context destroyed)
 * @return Status code
 */
imp_status_t imp_context_get_device_name(imp_context_t* ctx,
                                          const char** device_name);

/**
 * Destroy context
 * 
 * @param ctx Context handle
 */
void imp_context_destroy(imp_context_t* ctx);

// TODO: imp_context_create_from_device(imp_context_t** ctx, const char* device_name)
// Creates a pure media context for decode/encode without a model.
// Needed for transcode workflows with no inference.

/**
 * Get last error message for context
 * 
 * @param ctx Context handle
 * @return Error message string (valid until next API call)
 */
const char* imp_context_get_error(imp_context_t* ctx);

//////////////////////////////////////////////////////////////////////////////
// Video Source Configuration
//////////////////////////////////////////////////////////////////////////////

typedef enum {
    IMP_SOURCE_FILE = 0,           // Local file path
    IMP_SOURCE_URL = 1,            // Network URL (rtsp://, http://, udp://)
    IMP_SOURCE_CAMERA = 2,         // Camera device
} imp_source_type_t;

/**
 * Opaque video source configuration handle (pointer type)
 */
typedef struct imp_video_source_s imp_video_source_t;

/**
 * Create video source configuration
 * 
 * @param source Output: source configuration handle pointer
 * @param type Source type
 * @return Status code
 */
imp_status_t imp_video_source_create(imp_video_source_t** source,
                                     imp_source_type_t type);

/**
 * Set source property
 * 
 * Standard keys by source type:
 *   IMP_SOURCE_FILE:
 *     - "path": file path (required)
 *   IMP_SOURCE_URL:
 *     - "url": stream URL (required)
 *     - "transport": "tcp" or "udp" (RTSP only)
 *     - "username": auth username
 *     - "password": auth password
 *     - "timeout": connection timeout in ms
 *     - "low_latency": "1" to enable
 *   IMP_SOURCE_CAMERA:
 *     - "device": device index ("0", "1") or path ("/dev/video0")
 *     - "width": preferred width
 *     - "height": preferred height
 *     - "framerate": preferred FPS
 *     - "format": capture format ("mjpeg", "nv12", "yuyv")
 * 
 * @param source Source configuration handle
 * @param key Property key
 * @param value Property value (as string)
 * @return Status code
 */
imp_status_t imp_video_source_set(imp_video_source_t* source,
                                  const char* key,
                                  const char* value);

/**
 * Destroy source configuration
 * 
 * @param source Source configuration handle
 */
void imp_video_source_destroy(imp_video_source_t* source);

//////////////////////////////////////////////////////////////////////////////
// Decode Configuration
//////////////////////////////////////////////////////////////////////////////

/**
 * Image decode options
 */
typedef struct {
    imp_pixel_format_t output_format;   // Desired output format (default: RGB)
    imp_layout_t output_layout;         // Tensor layout (default: NHWC)
    imp_element_type_t output_type;     // Element type (default: U8)
    uint32_t resize_width;              // 0 = keep original
    uint32_t resize_height;             // 0 = keep original
    const char* decode_device;          // Where decoding runs: "CPU", "GPU", "GPU.0", "GPU.1", "NPU", NULL = match context
    const char* output_device;          // Where result tensor is placed: NULL = same as decode_device
} imp_image_decode_opts_t;

/**
 * Video decode output branch
 * 
 * Defines one output branch at a specific resolution/format.
 * Multiple branches share a single decode + tee pipeline on GPU.
 * Width/height of 0 means: deduce from context model dims.
 * If context has no model and dims are 0, source resolution is used.
 */
typedef struct {
    uint32_t width;                     // Output width  (0 = deduce from model, or source res)
    uint32_t height;                    // Output height (0 = deduce from model, or source res)
    imp_pixel_format_t format;          // Output format (default: NV12)
    const char* name;                   // Optional branch name (for imp_video_read_frame_by_name)
} imp_video_branch_t;

/**
 * Video stream configuration
 * 
 * If branches is NULL and branch_count is 0, a single branch at source
 * resolution is created automatically.
 */
typedef struct {
    imp_pixel_format_t output_format;   // Base format after decode (default: NV12)
    imp_layout_t output_layout;         // Tensor layout (default: NHWC)
    imp_element_type_t output_type;     // Element type (default: U8)
    const char* decode_device;          // Where decoding runs: "CPU", "GPU", "GPU.0", "GPU.1", "NPU", NULL = match context
    const char* output_device;          // Where result tensor is placed: NULL = same as decode_device
    uint32_t buffer_count;              // Frame buffer pool size (default: 4)
    int64_t timeout_ms;                 // Read timeout, -1 = infinite
    const imp_video_branch_t* branches; // Array of output branches (NULL = single branch at source res) // TODO FIXME do we need more than one?
    uint32_t branch_count;              // Number of branches (0 = single branch at source res)
    bool sync_appsink;                  // If true (file mode): force single-buffer queue+appsink (disable pre-buffering)
    uint32_t queue_depth;               // If nonzero (file mode): bound queue + appsink max-buffers to N (overrides sync_appsink)
} imp_video_decode_opts_t;

/**
 * Audio decode options
 */
typedef struct {
    uint32_t sample_rate;               // Target sample rate (0 = keep original)
    uint32_t channels;                  // Target channels (0 = keep original)
    imp_element_type_t output_type;     // Element type (default: FP32)
    bool normalize;                     // Normalize to [-1,1] for float types (default: true)
} imp_audio_decode_opts_t;

//////////////////////////////////////////////////////////////////////////////
// Encode Configuration
//////////////////////////////////////////////////////////////////////////////

/**
 * Image encode options
 */
typedef struct {
    const char* format;                 // "jpeg", "png", "bmp"
    int quality;                        // JPEG quality 1-100 (default: 90)
    const char* input_device;           // Expected input tensor location: NULL = accept any, copy if needed
    const char* encode_device;          // Where encoding runs: "CPU", "GPU", "GPU.0", etc., NULL = match context
} imp_image_encode_opts_t;

/**
 * Video encode options
 */
typedef struct {
    const char* codec;                  // "h264", "h265", "av1"
    uint32_t bitrate_kbps;              // Target bitrate
    uint32_t framerate;                 // Frames per second
    const char* input_device;           // Expected input tensor location: NULL = accept any, copy if needed
    const char* encode_device;          // Where encoding runs: "CPU", "GPU", "GPU.0", etc., NULL = match context
    const char* output_path;            // Output file path (NULL for memory)
} imp_video_encode_opts_t;

//////////////////////////////////////////////////////////////////////////////
// Async Callback Types
//////////////////////////////////////////////////////////////////////////////

/**
 * Decode completion callback
 * 
 * @param status Operation status
 * @param tensor Output tensor (owned by caller after callback)
 * @param user_data User-provided data
 */
typedef void (*imp_decode_callback_t)(imp_status_t status,
                                      imp_tensor_t* tensor,
                                      void* user_data);

/**
 * Video frame callback (for streaming)
 * 
 * @param status Operation status (IMP_ERROR_STREAM_END when done)
 * @param tensor Frame tensor (valid only during callback)
 * @param frame_index Frame number (0-based)
 * @param timestamp_us Presentation timestamp in microseconds
 * @param user_data User-provided data
 * @return true to continue, false to stop stream
 */
typedef bool (*imp_video_frame_callback_t)(imp_status_t status,
                                           imp_tensor_t* tensor,
                                           uint64_t frame_index,
                                           int64_t timestamp_us,
                                           void* user_data);

/**
 * Encode completion callback
 * 
 * @param status Operation status
 * @param data Encoded data (owned by caller after callback)
 * @param size Data size in bytes
 * @param user_data User-provided data
 */
typedef void (*imp_encode_callback_t)(imp_status_t status,
                                      void* data,
                                      size_t size,
                                      void* user_data);

//////////////////////////////////////////////////////////////////////////////
// Image Decode
//////////////////////////////////////////////////////////////////////////////

/**
 * Decode image from memory buffer
 * 
 * @param tensor Output: tensor handle pointer (CPU or GPU depending on context/options)
 * @param data Encoded image data (JPEG, PNG, etc.)
 * @param size Data size in bytes
 * @param ctx Context handle
 * @param opts Decode options (NULL for defaults)
 * @param callback Async callback (NULL for synchronous)
 * @param user_data User data for callback
 * @return Status code (IMP_OK if async started successfully)
 */
imp_status_t imp_decode_image(imp_tensor_t** tensor,
                              const void* data,
                              size_t size,
                              imp_context_t* ctx,
                              const imp_image_decode_opts_t* opts,
                              imp_decode_callback_t callback,
                              void* user_data);

/**
 * Decode image from file
 * 
 * @param tensor Output: tensor handle pointer (CPU or GPU depending on context/options)
 * @param file_path Path to image file
 * @param ctx Context handle
 * @param opts Decode options (NULL for defaults)
 * @param callback Async callback (NULL for synchronous)
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_decode_image_file(imp_tensor_t** tensor,
                                   const char* file_path,
                                   imp_context_t* ctx,
                                   const imp_image_decode_opts_t* opts,
                                   imp_decode_callback_t callback,
                                   void* user_data);

//////////////////////////////////////////////////////////////////////////////
// Video Decode
//////////////////////////////////////////////////////////////////////////////

/**
 * Opaque video stream handle (pointer type)
 */
typedef struct imp_video_stream_s imp_video_stream_t;

/**
 * Open video stream from source configuration
 * 
 * @param stream Output: stream handle pointer
 * @param source Source configuration (ownership transferred, do not destroy separately)
 * @param ctx Context handle
 * @param opts Stream options (NULL for defaults)
 * @return Status code
 */
imp_status_t imp_video_open(imp_video_stream_t** stream,
                            imp_video_source_t* source,
                            imp_context_t* ctx,
                            const imp_video_decode_opts_t* opts);

/**
 * Read next frame from a specific branch
 * 
 * @param tensor Output: frame tensor pointer (NV12 data, CPU side)
 * @param stream Stream handle
 * @param branch_index Branch index (0-based, must be < branch_count)
 * @return Status code (IMP_ERROR_STREAM_END when no more frames)
 */
imp_status_t imp_video_read_frame(imp_tensor_t** tensor,
                                  imp_video_stream_t* stream,
                                  uint32_t branch_index);

/**
 * Read next frame from a named branch
 * 
 * @param tensor Output: frame tensor pointer (NV12 data, CPU side)
 * @param stream Stream handle
 * @param branch_name Branch name (as specified in imp_video_branch_t)
 * @return Status code (IMP_ERROR_STREAM_END when no more frames)
 */
imp_status_t imp_video_read_frame_by_name(imp_tensor_t** tensor,
                                          imp_video_stream_t* stream,
                                          const char* branch_name);

/**
 * Start async frame processing with callback
 * 
 * @param stream Stream handle
 * @param callback Frame callback
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_video_start_async(imp_video_stream_t* stream,
                                   imp_video_frame_callback_t callback,
                                   void* user_data);

/**
 * Stop async processing
 * 
 * @param stream Stream handle
 */
void imp_video_stop(imp_video_stream_t* stream);

/**
 * Get video metadata
 * 
 * @param stream Stream handle
 * @param width Output: frame width
 * @param height Output: frame height
 * @param fps Output: frames per second
 * @param frame_count Output: total frames (-1 if unknown/live)
 * @return Status code
 */
imp_status_t imp_video_get_info(imp_video_stream_t* stream,
                                uint32_t* width,
                                uint32_t* height,
                                float* fps,
                                int64_t* frame_count);

/**
 * Close video stream
 * 
 * @param stream Stream handle
 */
void imp_video_close(imp_video_stream_t* stream);

/**
 * Query video file metadata
 * 
 * Use before decode to determine file properties.
 * 
 * @param file_path Path to video file
 * @param width Output: frame width (can be NULL)
 * @param height Output: frame height (can be NULL)
 * @param fps Output: frames per second (can be NULL)
 * @param frame_count Output: total frames, -1 if unknown (can be NULL)
 * @param duration_sec Output: duration in seconds (can be NULL)
 * @return Status code
 */
imp_status_t imp_video_file_info(const char* file_path,
                                 uint32_t* width,
                                 uint32_t* height,
                                 float* fps,
                                 int64_t* frame_count,
                                 double* duration_sec);

// TODO: One-shot video decode (batch decode entire video at once)
// Decode entire video to 4D tensor [frames, H, W, C]
// imp_status_t imp_decode_video(imp_tensor_t** tensor, const void* data, size_t size, ...);
// imp_status_t imp_decode_video_file(imp_tensor_t** tensor, const char* file_path, ...);

//////////////////////////////////////////////////////////////////////////////
// Audio Decode
//////////////////////////////////////////////////////////////////////////////

/**
 * Decode audio from memory buffer
 * 
 * @param tensor Output: tensor handle pointer with audio samples
 * @param data Encoded audio data
 * @param size Data size in bytes
 * @param ctx Context handle
 * @param opts Decode options (NULL for defaults)
 * @param callback Async callback (NULL for synchronous)
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_decode_audio(imp_tensor_t** tensor,
                              const void* data,
                              size_t size,
                              imp_context_t* ctx,
                              const imp_audio_decode_opts_t* opts,
                              imp_decode_callback_t callback,
                              void* user_data);

/**
 * Decode audio from file
 * 
 * @param tensor Output: tensor handle pointer with audio samples
 * @param file_path Path to audio file
 * @param ctx Context handle
 * @param opts Decode options (NULL for defaults)
 * @param callback Async callback (NULL for synchronous)
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_decode_audio_file(imp_tensor_t** tensor,
                                   const char* file_path,
                                   imp_context_t* ctx,
                                   const imp_audio_decode_opts_t* opts,
                                   imp_decode_callback_t callback,
                                   void* user_data);

/**
 * Query audio file metadata
 * 
 * Use before decode to determine file properties and allocate appropriately.
 * 
 * @param file_path Path to audio file
 * @param sample_rate Output: sample rate in Hz (can be NULL)
 * @param channels Output: number of channels (can be NULL)
 * @param duration_sec Output: duration in seconds (can be NULL)
 * @return Status code
 */
imp_status_t imp_audio_file_info(const char* file_path,
                                 uint32_t* sample_rate,
                                 uint32_t* channels,
                                 double* duration_sec);

//////////////////////////////////////////////////////////////////////////////
// Audio Stream (high-level decode → encode pipeline)
//////////////////////////////////////////////////////////////////////////////

/**
 * Audio stream options
 */
typedef struct {
    uint32_t    sample_rate;            // Output sample rate (0 = 44100)
    uint32_t    channels;               // Output channels (0 = 2)
    const char* output_codec;           // "mp3","aac","flac","wav","opus" (NULL = "mp3")
    uint32_t    output_bitrate_kbps;    // Bitrate for lossy codecs (0 = 192)
    bool        expose_samples;         // Reserved for future use
} imp_audio_stream_opts_t;

/**
 * Audio stream info (returned by imp_audio_get_info)
 */
typedef struct {
    uint32_t sample_rate;
    uint32_t channels;
    double   duration_sec;
    int64_t  num_samples;
} imp_audio_info_t;

/**
 * Opaque audio stream handle (pointer type)
 */
typedef struct imp_audio_stream_s imp_audio_stream_t;

/**
 * Open audio stream for processing
 * 
 * Creates a decode → encode pipeline for file-based audio processing.
 * 
 * @param stream Output: stream handle pointer
 * @param input_path Input audio file path
 * @param output_path Output audio file path (NULL for decode-only)
 * @param opts Stream options (NULL for defaults: 44100 Hz, stereo, MP3 192kbps)
 * @return Status code
 */
imp_status_t imp_audio_open(imp_audio_stream_t** stream,
                            const char* input_path,
                            const char* output_path,
                            const imp_audio_stream_opts_t* opts);

/**
 * Get audio stream info (duration, sample rate, channels)
 * 
 * @param stream Stream handle
 * @param info Output: audio info structure
 * @return Status code
 */
imp_status_t imp_audio_get_info(imp_audio_stream_t* stream,
                                imp_audio_info_t* info);

/**
 * Process audio stream (run decode → encode pipeline to completion)
 * 
 * @param stream Stream handle
 * @return Status code
 */
imp_status_t imp_audio_process(imp_audio_stream_t* stream);

/**
 * Get processing timing information
 * 
 * @param stream Stream handle
 * @param wall_time_sec Output: wall-clock time in seconds
 * @param realtime_factor Output: audio_duration / wall_time (>1 = faster than realtime)
 * @return Status code
 */
imp_status_t imp_audio_get_timing(imp_audio_stream_t* stream,
                                  double* wall_time_sec,
                                  double* realtime_factor);

/**
 * Close audio stream and free resources
 * 
 * @param stream Stream handle
 */
void imp_audio_close(imp_audio_stream_t* stream);

//////////////////////////////////////////////////////////////////////////////
// Image Encode
//////////////////////////////////////////////////////////////////////////////

/**
 * Encode tensor to image format
 * 
 * @param data Output: encoded data (caller must free with imp_free)
 * @param size Output: data size
 * @param tensor Input tensor
 * @param ctx Context handle
 * @param opts Encode options (NULL for defaults: JPEG quality 90)
 * @param callback Async callback (NULL for synchronous)
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_encode_image(void** data,
                              size_t* size,
                              imp_tensor_t* tensor,
                              imp_context_t* ctx,
                              const imp_image_encode_opts_t* opts,
                              imp_encode_callback_t callback,
                              void* user_data);

/**
 * Encode tensor to image file
 * 
 * @param file_path Output file path (format inferred from extension)
 * @param tensor Input tensor
 * @param ctx Context handle
 * @param opts Encode options (NULL for defaults)
 * @param callback Async callback (NULL for synchronous)
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_encode_image_file(const char* file_path,
                                   imp_tensor_t* tensor,
                                   imp_context_t* ctx,
                                   const imp_image_encode_opts_t* opts,
                                   imp_encode_callback_t callback,
                                   void* user_data);

//////////////////////////////////////////////////////////////////////////////
// Video Encode
//////////////////////////////////////////////////////////////////////////////

/**
 * Opaque video encoder handle (pointer type)
 */
typedef struct imp_video_encoder_s imp_video_encoder_t;

/**
 * Create video encoder
 * 
 * @param encoder Output: encoder handle pointer
 * @param width Frame width
 * @param height Frame height
 * @param ctx Context handle
 * @param opts Encode options
 * @return Status code
 */
imp_status_t imp_video_encoder_create(imp_video_encoder_t** encoder,
                                      uint32_t width,
                                      uint32_t height,
                                      imp_context_t* ctx,
                                      const imp_video_encode_opts_t* opts);

/**
 * Encode frame
 * 
 * @param encoder Encoder handle
 * @param tensor Frame tensor
 * @return Status code
 */
imp_status_t imp_video_encoder_write(imp_video_encoder_t* encoder,
                                     imp_tensor_t* tensor);

/**
 * Finalize and close encoder
 * 
 * @param encoder Encoder handle
 */
void imp_video_encoder_close(imp_video_encoder_t* encoder);

// TODO: One-shot video encode (batch encode entire video at once)
// Encode 4D tensor [frames, H, W, C] to video
// To memory (e.g. for network send):
// imp_status_t imp_encode_video(void** data, size_t* size, imp_tensor_t* tensor, ...);
// To file:
// imp_status_t imp_encode_video_file(const char* file_path, imp_tensor_t* tensor, ...);

//////////////////////////////////////////////////////////////////////////////
// Audio Encode
//////////////////////////////////////////////////////////////////////////////

/**
 * Audio encode options
 */
typedef struct {
    const char* codec;                  // "mp3", "aac", "opus", "flac", "wav"
    uint32_t bitrate_kbps;              // Target bitrate for lossy codecs (default: 192)
    uint32_t sample_rate;               // Output sample rate (0 = match input)
    uint32_t channels;                  // Output channels (0 = match input)
    const char* output_path;            // Output file path
} imp_audio_encode_opts_t;

/**
 * Opaque audio encoder handle (pointer type)
 */
typedef struct imp_audio_encoder_s imp_audio_encoder_t;

/**
 * Create audio encoder
 * 
 * @param encoder Output: encoder handle pointer
 * @param opts Encode options
 * @param callback Completion callback (NULL for synchronous close)
 * @param user_data User data for callback
 * @return Status code
 */
imp_status_t imp_audio_encoder_create(imp_audio_encoder_t** encoder,
                                      const imp_audio_encode_opts_t* opts,
                                      imp_encode_callback_t callback,
                                      void* user_data);

/**
 * Write audio samples to encoder
 * 
 * @param encoder Encoder handle
 * @param tensor Audio samples tensor (from imp_decode_audio_file or custom)
 * @return Status code
 */
imp_status_t imp_audio_encoder_write(imp_audio_encoder_t* encoder,
                                     imp_tensor_t* tensor);

/**
 * Finalize and close encoder
 * 
 * If callback was provided at create, it will be called when encoding completes.
 * 
 * @param encoder Encoder handle
 */
void imp_audio_encoder_close(imp_audio_encoder_t* encoder);

/**
 * One-shot audio encode to memory buffer.
 *
 * Encodes raw float PCM samples into the specified codec format and
 * returns the encoded bytes.  For "wav" the library uses a built-in
 * header writer (no GStreamer overhead).  For "pcm" the raw float
 * bytes are returned as-is.  For lossy/lossless codecs ("mp3",
 * "flac", "opus", "aac") a GStreamer pipeline is created internally.
 *
 * The caller must free the returned buffer with imp_free().
 *
 * @param data        Output: pointer to encoded data (heap-allocated)
 * @param data_size   Output: size of encoded data in bytes
 * @param samples     Input float PCM samples (mono, interleaved if stereo)
 * @param num_samples Number of float values in @p samples
 * @param opts        Encode options (codec, sample_rate, channels, bitrate).
 *                    opts->output_path is ignored - output goes to memory.
 * @return IMP_OK on success
 */
imp_status_t imp_encode_audio(void** data,
                              size_t* data_size,
                              const float* samples,
                              size_t num_samples,
                              const imp_audio_encode_opts_t* opts);

/**
 * One-shot audio encode to file.
 *
 * Same as imp_encode_audio() but writes directly to disk.
 *
 * @param file_path   Output file path
 * @param samples     Input float PCM samples
 * @param num_samples Number of float values
 * @param opts        Encode options (codec, bitrate, sample_rate, channels).
 *                    opts->output_path is overridden by @p file_path.
 * @return IMP_OK on success
 */
imp_status_t imp_encode_audio_file(const char* file_path,
                                   const float* samples,
                                   size_t num_samples,
                                   const imp_audio_encode_opts_t* opts);

//////////////////////////////////////////////////////////////////////////////
// Tensor Utilities
//////////////////////////////////////////////////////////////////////////////

/**
 * Get tensor's device type
 * 
 * @param tensor Tensor handle
 * @param device_type Output: device type (CPU, GPU, NPU)
 * @return Status code
 */
imp_status_t imp_tensor_get_device_type(imp_tensor_t* tensor,
                                         imp_device_type_t* device_type);

/**
 * Get tensor's device name (e.g., "GPU.0", "GPU.1", "CPU", "NPU")
 * 
 * @param tensor Tensor handle
 * @param device_name Output: device name string
 * @return Status code
 */
imp_status_t imp_tensor_get_device_name(imp_tensor_t* tensor,
                                         const char** device_name);

/**
 * Get tensor's context type (for GPU/NPU tensors)
 * Returns IMP_ERROR_INVALID_ARGUMENT for CPU tensors.
 * 
 * @param tensor Tensor handle
 * @param context_type Output: context type (OpenCL, D3D11, VA-API)
 * @return Status code
 */
imp_status_t imp_tensor_get_context_type(imp_tensor_t* tensor,
                                          imp_context_type_t* context_type);

/**
 * Get underlying OpenVINO tensor pointer
 * 
 * @param tensor Tensor handle
 * @param ov_tensor Output: pointer to underlying OV tensor (ov::Tensor* or ov::RemoteTensor*)
 * @param device_type Output: device type (to know actual type for casting)
 * @return Status code
 */
imp_status_t imp_tensor_get_ov(imp_tensor_t* tensor,
                                void** ov_tensor,
                                imp_device_type_t* device_type);

/**
 * Get tensor shape
 * 
 * @param tensor Tensor handle
 * @param dims Output: dimension array (caller provides)
 * @param num_dims Input: array size, Output: actual dimensions
 * @return Status code
 */
imp_status_t imp_tensor_get_shape(imp_tensor_t* tensor,
                                  int64_t* dims,
                                  size_t* num_dims);

/**
 * Get tensor element type
 * 
 * @param tensor Tensor handle
 * @param type Output: element type
 * @return Status code
 */
imp_status_t imp_tensor_get_element_type(imp_tensor_t* tensor,
                                          imp_element_type_t* type);

/**
 * Get NV12 plane pointers and dimensions for CPU-side NV12 tensors.
 *
 * @param tensor     Tensor handle (must be format IMP_FORMAT_NV12, device CPU)
 * @param y_data     Output: pointer to Y plane (width * height bytes)
 * @param uv_data    Output: pointer to interleaved UV plane (width * height/2 bytes)
 * @param width      Output: frame width in pixels
 * @param height     Output: frame height in pixels
 * @return IMP_OK on success, IMP_ERROR_INVALID_ARGUMENT if tensor is not NV12/CPU
 */
imp_status_t imp_tensor_get_nv12_planes(imp_tensor_t* tensor,
                                        const uint8_t** y_data,
                                        const uint8_t** uv_data,
                                        int* width,
                                        int* height);

/**
 * Release tensor
 * 
 * @param tensor Tensor handle
 */
void imp_tensor_release(imp_tensor_t* tensor);

//////////////////////////////////////////////////////////////////////////////
// Memory Management
//////////////////////////////////////////////////////////////////////////////

/**
 * Free memory allocated by IMP functions
 * 
 * @param ptr Pointer to free
 */
void imp_free(void* ptr);

//////////////////////////////////////////////////////////////////////////////
// Utility
//////////////////////////////////////////////////////////////////////////////

/**
 * Get API version
 * 
 * @param major Output: major version
 * @param minor Output: minor version  
 * @param patch Output: patch version
 */
void imp_get_version(int* major, int* minor, int* patch);

/**
 * Check hardware decode support
 * 
 * @param ctx Context handle
 * @param supported Output: true if HW decode available
 * @return Status code
 */
imp_status_t imp_hw_decode_supported(imp_context_t* ctx, bool* supported);

/**
 * Check hardware encode support
 * 
 * @param ctx Context handle
 * @param supported Output: true if HW encode available
 * @return Status code
 */
imp_status_t imp_hw_encode_supported(imp_context_t* ctx, bool* supported);

#ifdef __cplusplus
}
#endif

#endif // INTEL_MPI_H
