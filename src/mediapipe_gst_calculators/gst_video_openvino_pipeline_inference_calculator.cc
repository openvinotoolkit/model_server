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

#include <cstdlib>
#include <chrono>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#pragma GCC diagnostic pop

#include "src/mediapipe_gst_calculators/gst_openvino_inference_calculator.pb.h"
#include "src/mpi/intel_mpi.h"

#ifdef _WIN32
#ifndef GST_USE_UNSTABLE_API
#define GST_USE_UNSTABLE_API
#endif
#include <d3d11.h>
#include <gst/base/gstbasetransform.h>
#include <gst/d3d11/gstd3d11.h>
#include <wrl/client.h>
#endif

namespace {
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kMaxFramesTag[] = "MAX_FRAMES";
constexpr char kFramesTag[] = "FRAMES";

#ifdef _WIN32

struct OpenVinoState {
    ov::Core core;
    ov::CompiledModel compiled_model;
    std::vector<ov::InferRequest> infer_requests;
    std::unique_ptr<ov::intel_gpu::ocl::D3DContext> d3d_context;
    size_t request_index = 0;
};

typedef struct _GstOvD3D11Inference {
    GstBaseTransform parent;
    gchar* model_path;
    gchar* input_name;
    gchar* performance_hint;
    guint num_streams;
    gint width;
    gint height;
    gboolean copy_input;
    gboolean diagnostics;
    guint request_count;
    guint64 infer_count;
    double infer_ms_total;
    D3D11_TEXTURE2D_DESC imported_texture_desc;
    gboolean has_imported_texture_desc;
    OpenVinoState* state;
} GstOvD3D11Inference;

typedef struct _GstOvD3D11InferenceClass {
    GstBaseTransformClass parent_class;
} GstOvD3D11InferenceClass;

#define GST_TYPE_OV_D3D11_INFERENCE (gst_ov_d3d11_inference_get_type())
#define GST_OV_D3D11_INFERENCE(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_OV_D3D11_INFERENCE, GstOvD3D11Inference))

G_DEFINE_TYPE(GstOvD3D11Inference, gst_ov_d3d11_inference, GST_TYPE_BASE_TRANSFORM)

enum {
    kPropertyZero,
    kPropertyModelPath,
    kPropertyInputName,
    kPropertyPerformanceHint,
    kPropertyNumStreams,
};

static gboolean initializeInference(GstOvD3D11Inference* self, ID3D11Device* device) {
    try {
        if (self->model_path == nullptr || *self->model_path == '\0' ||
            self->input_name == nullptr || *self->input_name == '\0' ||
            self->width <= 0 || self->height <= 0 || device == nullptr)
            return FALSE;
        self->state = new OpenVinoState();
        std::shared_ptr<ov::Model> model = self->state->core.read_model(
            self->model_path, {}, ov::AnyMap{{"ENABLE_MMAP", "NO"}});
        ov::preprocess::PrePostProcessor preprocessor(model);
        preprocessor.input().tensor().set_element_type(ov::element::u8)
            .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
            .set_memory_type(ov::intel_gpu::memory_type::surface);
        preprocessor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        preprocessor.input().model().set_layout("NCHW");
        self->state->d3d_context = std::make_unique<ov::intel_gpu::ocl::D3DContext>(
            self->state->core, device);
        const ov::AnyMap config{{"PERFORMANCE_HINT", std::string(self->performance_hint)},
                                {"NUM_STREAMS", std::to_string(self->num_streams)}};
        self->state->compiled_model = self->state->core.compile_model(
            preprocessor.build(), *self->state->d3d_context, config);
        self->state->infer_requests.reserve(self->request_count);
        for (guint request_index = 0; request_index < self->request_count; ++request_index)
            self->state->infer_requests.emplace_back(self->state->compiled_model.create_infer_request());
        return TRUE;
    } catch (const std::exception& exception) {
        g_printerr("ovd3d11infer initialization failed: %s\n", exception.what());
        delete self->state;
        self->state = nullptr;
        return FALSE;
    }
}

static GstFlowReturn gstOvD3D11InferenceTransformIp(GstBaseTransform* transform, GstBuffer* buffer) {
    GstOvD3D11Inference* self = GST_OV_D3D11_INFERENCE(transform);
    GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
    if (memory == nullptr || !gst_is_d3d11_memory(memory)) {
        g_printerr("ovd3d11infer requires D3D11 memory\n");
        return GST_FLOW_NOT_NEGOTIATED;
    }
    GstD3D11Memory* d3d_memory = reinterpret_cast<GstD3D11Memory*>(memory);
    ID3D11Resource* resource = gst_d3d11_memory_get_resource_handle(d3d_memory);
    ID3D11Texture2D* texture = static_cast<ID3D11Texture2D*>(resource);
    ID3D11Device* device = d3d_memory->device
        ? gst_d3d11_device_get_device_handle(d3d_memory->device)
        : nullptr;
    if (texture == nullptr || device == nullptr ||
        gst_d3d11_memory_get_subresource_index(d3d_memory) != 0) {
        g_printerr("ovd3d11infer received an unsupported D3D11 texture\n");
        return GST_FLOW_ERROR;
    }
    if (self->state == nullptr && !initializeInference(self, device)) {
        g_printerr("ovd3d11infer could not initialize OpenVINO\n");
        return GST_FLOW_ERROR;
    }
    try {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> copied_texture;
        ID3D11Texture2D* inference_texture = texture;
        gst_d3d11_device_lock(d3d_memory->device);
        ID3D11DeviceContext* device_context =
            gst_d3d11_device_get_device_context_handle(d3d_memory->device);
        if (self->copy_input) {
            D3D11_TEXTURE2D_DESC descriptor{};
            texture->GetDesc(&descriptor);
            descriptor.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
            descriptor.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
            descriptor.Usage = D3D11_USAGE_DEFAULT;
            descriptor.CPUAccessFlags = 0;
            if (FAILED(device->CreateTexture2D(&descriptor, nullptr, copied_texture.GetAddressOf()))) {
                gst_d3d11_device_unlock(d3d_memory->device);
                return GST_FLOW_ERROR;
            }
            device_context->CopyResource(copied_texture.Get(), texture);
            inference_texture = copied_texture.Get();
        }
        if (self->diagnostics && !self->has_imported_texture_desc) {
            inference_texture->GetDesc(&self->imported_texture_desc);
            self->has_imported_texture_desc = TRUE;
        }
        device_context->Flush();
        gst_d3d11_device_unlock(d3d_memory->device);
        const ov::Shape shape_y{1, static_cast<size_t>(self->height), static_cast<size_t>(self->width), 1};
        const ov::Shape shape_uv{1, static_cast<size_t>(self->height / 2), static_cast<size_t>(self->width / 2), 2};
        ov::InferRequest& infer_request = self->state->infer_requests[
            self->state->request_index++ % self->state->infer_requests.size()];
        infer_request.set_tensor(std::string(self->input_name) + "/y",
            self->state->d3d_context->create_tensor(ov::element::u8, shape_y, inference_texture, 0));
        infer_request.set_tensor(std::string(self->input_name) + "/uv",
            self->state->d3d_context->create_tensor(ov::element::u8, shape_uv, inference_texture, 1));
        const auto infer_start = self->diagnostics ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
        infer_request.infer();
        if (self->diagnostics) {
            self->infer_ms_total += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - infer_start).count();
            ++self->infer_count;
        }
        const ov::Tensor output = infer_request.get_output_tensor(0);
        if (output.data<const float>() == nullptr)
            return GST_FLOW_ERROR;
    } catch (const std::exception& exception) {
        g_printerr("ovd3d11infer inference failed: %s\n", exception.what());
        return GST_FLOW_ERROR;
    }
    return GST_FLOW_OK;
}

static gboolean gstOvD3D11InferenceSetCaps(GstBaseTransform* transform, GstCaps* caps, GstCaps*) {
    GstOvD3D11Inference* self = GST_OV_D3D11_INFERENCE(transform);
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    return gst_structure_get_int(structure, "width", &self->width) &&
           gst_structure_get_int(structure, "height", &self->height);
}

static void gstOvD3D11InferenceSetProperty(GObject* object, guint property_id,
                                            const GValue* value, GParamSpec*) {
    GstOvD3D11Inference* self = GST_OV_D3D11_INFERENCE(object);
    switch (property_id) {
    case kPropertyModelPath:
        g_free(self->model_path);
        self->model_path = g_value_dup_string(value);
        break;
    case kPropertyInputName:
        g_free(self->input_name);
        self->input_name = g_value_dup_string(value);
        break;
    case kPropertyPerformanceHint:
        g_free(self->performance_hint);
        self->performance_hint = g_value_dup_string(value);
        break;
    case kPropertyNumStreams:
        self->num_streams = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, nullptr);
    }
}

static void gstOvD3D11InferenceFinalize(GObject* object) {
    GstOvD3D11Inference* self = GST_OV_D3D11_INFERENCE(object);
    if (self->diagnostics && self->has_imported_texture_desc) {
        g_printerr("ovd3d11infer diagnostics mode=%s bind_flags=%u misc_flags=%u usage=%u format=%u "
                   "infer_count=%" G_GUINT64_FORMAT " infer_mean_ms=%.3f\n",
            self->copy_input ? "shared_copy" : "live", self->imported_texture_desc.BindFlags,
            self->imported_texture_desc.MiscFlags, self->imported_texture_desc.Usage,
            self->imported_texture_desc.Format, self->infer_count,
            self->infer_count == 0 ? 0.0 : self->infer_ms_total / self->infer_count);
    }
    delete self->state;
    g_free(self->model_path);
    g_free(self->input_name);
    g_free(self->performance_hint);
    G_OBJECT_CLASS(gst_ov_d3d11_inference_parent_class)->finalize(object);
}

static void gst_ov_d3d11_inference_class_init(GstOvD3D11InferenceClass* klass) {
    GObjectClass* object_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    object_class->set_property = gstOvD3D11InferenceSetProperty;
    object_class->finalize = gstOvD3D11InferenceFinalize;
    g_object_class_install_property(object_class, kPropertyModelPath,
        g_param_spec_string("model-path", "Model path", "OpenVINO XML model path", nullptr,
                            static_cast<GParamFlags>(G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY)));
    g_object_class_install_property(object_class, kPropertyInputName,
        g_param_spec_string("input-name", "Input name", "OpenVINO model input name", "data",
                            static_cast<GParamFlags>(G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY)));
    g_object_class_install_property(object_class, kPropertyPerformanceHint,
        g_param_spec_string("performance-hint", "Performance hint", "OpenVINO performance hint", "LATENCY",
                            static_cast<GParamFlags>(G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY)));
    g_object_class_install_property(object_class, kPropertyNumStreams,
        g_param_spec_uint("num-streams", "Number of streams", "OpenVINO NUM_STREAMS", 1, G_MAXUINT, 1,
                          static_cast<GParamFlags>(G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY)));
    gst_element_class_set_static_metadata(element_class, "OpenVINO D3D11 inference",
        "Filter/Video", "Runs OpenVINO on D3D11 NV12 buffers", "Intel");
    GstCaps* caps = gst_caps_from_string("video/x-raw(memory:D3D11Memory),format=NV12");
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, gst_caps_ref(caps)));
    gst_caps_unref(caps);
    transform_class->set_caps = gstOvD3D11InferenceSetCaps;
    transform_class->transform_ip = gstOvD3D11InferenceTransformIp;
}

static void gst_ov_d3d11_inference_init(GstOvD3D11Inference* self) {
    self->num_streams = 1;
    const char* copy_input = std::getenv("OVMS_GST_ELEMENT_COPY_INPUT");
    self->copy_input = copy_input != nullptr && std::atoi(copy_input) != 0;
    const char* diagnostics = std::getenv("OVMS_GST_ELEMENT_DIAGNOSTICS");
    self->diagnostics = diagnostics != nullptr && std::atoi(diagnostics) != 0;
    self->infer_count = 0;
    self->infer_ms_total = 0.0;
    self->has_imported_texture_desc = FALSE;
    self->request_count = 1;
    const char* request_count = std::getenv("OVMS_GST_ELEMENT_REQUEST_COUNT");
    if (request_count != nullptr) {
        const int parsed_request_count = std::atoi(request_count);
        if (parsed_request_count > 0)
            self->request_count = static_cast<guint>(parsed_request_count);
    }
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

bool registerOvD3D11InferenceElement() {
    static std::once_flag registration_once;
    static bool registered = false;
    std::call_once(registration_once, [] {
        registered = gst_element_register(nullptr, "ovd3d11infer", GST_RANK_NONE,
            GST_TYPE_OV_D3D11_INFERENCE);
    });
    return registered;
}

#endif
}  // namespace

namespace mediapipe {

class GstVideoOpenVinoPipelineInferenceCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->InputSidePackets().Tag(kVideoPathTag).Set<std::string>();
        cc->InputSidePackets().Tag(kWidthTag).Set<int>();
        cc->InputSidePackets().Tag(kHeightTag).Set<int>();
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            cc->InputSidePackets().Tag(kMaxFramesTag).Set<int>();
        cc->Outputs().Tag(kFramesTag).Set<int>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
#ifndef _WIN32
        return absl::UnimplementedError("GstVideoOpenVinoPipelineInferenceCalculator supports Windows D3D11 only");
#else
    gst_init(nullptr, nullptr);
        if (!registerOvD3D11InferenceElement())
            return absl::InternalError("could not register ovd3d11infer GStreamer element");
        const GstOpenVinoInferenceCalculatorOptions& options =
            cc->Options<GstOpenVinoInferenceCalculatorOptions>();
        if (options.model_path().empty() || options.input_name().empty() || options.num_streams() == 0)
            return absl::InvalidArgumentError("invalid OpenVINO pipeline inference options");
        if (cc->InputSidePackets().HasTag(kMaxFramesTag))
            max_frames_ = cc->InputSidePackets().Tag(kMaxFramesTag).Get<int>();
        int queue_depth = 4;
        const char* configured_queue_depth = std::getenv("OVMS_GST_ELEMENT_QUEUE_DEPTH");
        if (configured_queue_depth != nullptr) {
            const int parsed_queue_depth = std::atoi(configured_queue_depth);
            if (parsed_queue_depth >= 0)
                queue_depth = parsed_queue_depth;
        }
        filter_ = "queue max-size-buffers=" + std::to_string(queue_depth) +
            " max-size-time=0 max-size-bytes=0 ! "
            "ovd3d11infer model-path=\"" + options.model_path() + "\" input-name=\"" +
            options.input_name() + "\" performance-hint=\"" + options.performance_hint() +
            "\" num-streams=" + std::to_string(options.num_streams());
        if (imp_context_create(&context_, nullptr) != IMP_OK)
            return absl::InternalError("imp_context_create failed");
        imp_video_source_t* source = nullptr;
        if (imp_video_source_create(&source, IMP_SOURCE_FILE) != IMP_OK)
            return absl::InternalError("imp_video_source_create failed");
        const std::string video_path = cc->InputSidePackets().Tag(kVideoPathTag).Get<std::string>();
        const int width = cc->InputSidePackets().Tag(kWidthTag).Get<int>();
        const int height = cc->InputSidePackets().Tag(kHeightTag).Get<int>();
        imp_video_source_set(source, "path", video_path.c_str());
        imp_video_source_set(source, "width", std::to_string(width).c_str());
        imp_video_source_set(source, "height", std::to_string(height).c_str());
        imp_video_decode_opts_t decode_options{};
        decode_options.use_d3d11_surface_memory = true;
        decode_options.d3d11_filter = filter_.c_str();
        const imp_status_t status = imp_video_open(&stream_, source, context_, &decode_options);
        imp_video_source_destroy(source);
        if (status != IMP_OK)
            return absl::InternalError(imp_context_get_error(context_));
        return absl::OkStatus();
#endif
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (max_frames_ > 0 && frame_index_ >= max_frames_)
            return tool::StatusStop();
        imp_tensor_t* tensor = nullptr;
        const imp_status_t status = imp_video_read_frame(&tensor, stream_, 0);
        if (status == IMP_ERROR_STREAM_END)
            return tool::StatusStop();
        if (status != IMP_OK || tensor == nullptr)
            return absl::InternalError("pipeline inference frame read failed");
        imp_tensor_release(tensor);
        cc->Outputs().Tag(kFramesTag).AddPacket(MakePacket<int>(1).At(Timestamp(frame_index_++)));
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext*) override {
        if (stream_ != nullptr)
            imp_video_close(stream_);
        if (context_ != nullptr)
            imp_context_destroy(context_);
        return absl::OkStatus();
    }

 private:
    imp_context_t* context_ = nullptr;
    imp_video_stream_t* stream_ = nullptr;
    std::string filter_;
    int max_frames_ = 0;
    int64_t frame_index_ = 0;
};

REGISTER_CALCULATOR(GstVideoOpenVinoPipelineInferenceCalculator);

}  // namespace mediapipe