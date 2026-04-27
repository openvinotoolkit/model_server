#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
from collections import defaultdict
from itertools import chain
from typing import Type

import numpy as np

import tests.functional.constants.models
from tests.functional.utils.core import get_children_from_module
from ovms.config import language_models_enabled, ovms_types
from tests.functional.constants.models import (
    models_llm_vlm_ov_hf,
    models_llm_vlm_npu_ov_hf,
    AgeGender,
    Alexnet,
    AlibabaNLPGteLargeEnv15,
    ArgMax,
    AspireTdnn,
    BasicLstm,
    BAAIBgeBaseEnv15Fp16OvHf,
    BAAIBgeBaseEnv15Int8OvHf,
    BAAIBgeLargeZhv15,
    BAAIBgeLargeEnv15,
    BAAIRBgeRerankerBaseFp16OvHf,
    BAAIRBgeRerankerBaseInt8OvHf,
    BAAIRerankerLarge,
    BAAIRerankerV2M3,
    BAAIRerankerBase,
    Brain,
    Caffenet,
    CrnnTf,
    CrossEncoderMsmarcoMiniLML6EnDeV1,
    DeepSeekR1DistillQwen15BInt8,
    Densenet121,
    DistilWhisperLargeV3Int8OvHf,
    DLBertSmallUncasedFp32,
    DLBertSmallUncasedInt8,
    DLDeeplabV3Fp32,
    DLDeeplabV3Int8,
    DLDensenet121Fp32,
    DLDensenet121Int8,
    DLEfficientnetD0Fp32,
    DLEfficientnetD0Int8,
    DLGoogleNetV4Fp32,
    DLGoogleNetV4Int8,
    DLMobileNetSSDFp32,
    DLMobileNetSSDInt8,
    DLMobileNetV2Fp32,
    DLMobileNetV2Int8,
    DLResnet18Fp32,
    DLResnet18Int8,
    DLResnet50Fp32,
    DLResnet50Int8New,
    DLSSDResnet34Fp32,
    DLSSDResnet34Int8,
    DLUnetCamvidFp32,
    DLUnetCamvidInt8,
    DLYolo4Fp32,
    DLYolo4Int8,
    DLYoloV3TinyFp32,
    DLYoloV3TinyInt8,
    DreamlikeAnime10Int8,
    DreamlikeAnime10Int4,
    DreamlikeAnime10Int4SymCw,
    DreamlikeDiffusion10InpaintingInt8,
    DreamlikeDiffusion10InpaintingInt4,
    DreamlikeDiffusion10InpaintingInt4SymCw,
    Dreamshaper8InpaintingInt8OvHf,
    DummyAdd2Inputs,
    DummyIncrement,
    DummyIncrementDecrement,
    DummySavedModel,
    EastFp32,
    Emotion,
    FacebookOpt125Int8,
    Flux1SchnellInt4OvHf,
    Flux1SchnellInt8OvHf,
    TinyLlama11BChatV10Int4SymCw,
    FaceDetection,
    FaceDetectionRetail,
    GoogleNetV2Fp32,
    GptOss20BInt4OvHf,
    GptOss20BInt8OvHf,
    Gemma34BItInt4SymCw,
    Gemma34bItInt8OvHf,
    Gemma34bItInt4CwOvHf,
    Hermes3Llama318BInt8,
    Hermes3Llama318BInt4SymCw,
    Hermes3Llama318BInt4,
    InceptionResnetV2,
    Increment4d,
    InstanceSegmentationSecurity,
    InternVL21BInt8,
    InternVL21BInt4,
    InternVL22BInt8OvHf,
    IntfloatMultilingualE5LargeInstruct,
    IntfloatMultilingualE5Large,
    LCMDreamshaperv7Int8OvHf,
    Llama27BChatHfInt8,
    Llama27BChatHfInt4,
    Llama27BChatHfInt4SymCw,
    Llama323BInstructQ4KMGGUFInt4Hf,
    Llava157bHfInt4,
    Llava157bHfInt8,
    LSpeechV10,
    Matmul,
    MetaLlama318BInt8,
    MetaLlama318BInt4,
    MetaLlama318BInt4SymCw,
    MetaLlama318BInstructInt8,
    MetaLlama318BInstructInt4,
    MetaLlama318BInstructInt4SymCw,
    MetaLlama323BInstructInt8,
    MetaLlama323BInstructInt4,
    MetaLlama323BInstructInt4SymCw,
    MicrosoftSpeech5TtsFp16,
    MicrosoftSpeech5TtsInt8,
    Sandiago21Speech5TtsSpanishFp16,
    MiniCPMV26Int8,
    MiniCPMV26Int4,
    Phi35VisionInstructInt4SymCw,
    Phi35VisionInstructInt8OvHf,
    Mistral7BInstructv03Int8,
    Mistral7BInstructv03Int4,
    Mistral7BInstructv03Int8OvHf,
    Mistral7BInstructv03Int4CwOvHf,
    Mistral7BInstructv03Int4SymCw,
    Mnist,
    Muse,
    NoModel,
    NotExistingModelPath,
    NomicEmbedTextv15,
    MixedBreadAIDeepsetMxbaiEmbedLargeV1,
    OcrNetHrNetW48Paddle,
    OcrNetHrNetW48PaddleNative,
    Passthrough,
    Phi4MiniInstructInt8,
    Phi4MiniInstructInt4SymCw,
    Phi4MiniInstructInt4,
    Qwen205BInt4OvHf,
    Qwen2VL7BInstructInt8,
    Qwen2VL7BInstructInt4,
    Qwen3Coder30BA3BInt8OvHf,
    Qwen3Coder30BA3BInt4OvHf,
    Qwen306BInstructGGUFInt8Hf,
    Qwen3Embedding06B,
    Qwen3Embedding06BFp16OvHf,
    Qwen3Embedding06BInt8OvHf,
    Qwen3Reranker06BSeqCls,
    Qwen3Reranker06BSeqClsFp16OvHf,
    Qwen38BInt4OvHf,
    Qwen25VL7BInstructInt8OvHf,
    Qwen3VL4BInstructInt8,
    Qwen3VL4BInstructInt4,
    Qwen3VL32BInstructInt4,
    Qwen3VL8BInt4OvHf,
    DevstralSmall2507Int4,
    RcnnIlsvrc13,
    Resnet,
    Resnet50Binary,
    ResnetModelNameWithSlash,
    ResnetModelNameWithWhitespace,
    RmLstm,
    ScalarDummy,
    SmolLM2135InstructGGUFFp16Hf,
    SentenceTransformersAllMpnetBaseV2,
    SentenceTransformersAllMiniLML12V2,
    SentenceTransformersMultiQaMpnetBaseDotV1,
    SentenceTransformersAllDistilrobertaV1,
    SsdliteMobilenetV2,
    StableDiffusionv15Int8OvHf,
    StableDiffusion35LargeTurboInt8,
    StableDiffusion35LargeTurboInt4,
    StableDiffusion35LargeTurboInt4SymCw,
    StableDiffusionInpaintingInt8,
    StableDiffusionInpaintingInt4,
    StableDiffusionInpaintingInt4SymCw,
    StableDiffusionXl10Inpainting01Int8,
    StableDiffusionXl10Inpainting01Int4,
    StableDiffusionXl10Inpainting01Int4SymCw,
    StableDiffusionXlBase10Int8,
    StableDiffusionXlBase10Int4,
    StableDiffusionXlBase10Int4SymCw,
    ThenlperGteSmall,
    UniversalSentenceEncoder,
    UnsupportedModel,
    VehicleAttributesRecognition,
    VehicleDetection,
    WhisperLargeV3Int4OvHf,
    WhisperLargeV3Int8OvHf,
    WhisperLargeV3Fp16OvHf,
    WhisperLargeV3Fp16,
    WhisperLargeV3Int8,
    WhisperLargeV3Int4,
    WhisperLargeV3TurboFp16,
    WhisperLargeV3TurboInt8,
    WhisperLargeV3TurboInt4,
    WhisperSmallFp16,
    WhisperSmallInt8,
    WrongExtensionModel,
    WrongXmlEmpty,
    WrongXmlSubdirectory,
    WrongXmlWithout,
    Yamnet,
    models_path,
    Qwen257BInstruct1MInt4SymCw,
    Qwen2VL7BInstructInt4SymCw,
    Qwen257BInstruct1MInt8,
    Qwen257BInstruct1MInt4,
    Qwen257BInstructInt4,
    Qwen257BInstructInt8,
    Qwen38BInt8,
    Qwen38BInt4,
    extra_acc_models,
    extra_test_models,
)
from ovms.constants.ov import OV
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.constants.ovms_type import OvmsType
from tfs.constants.models import TfsResnet50Fp32, TfsResnet50Int8, TfsUnet3D
from trt.constants.models import (
    BertLargeFp32,
    BertLargeInt8,
    BertSmallFp32,
    BertSmallInt8,
    Dummy,
    EfficientLite4,
    GoogleNet,
    MobileNet3Large,
    MobileNet3Small,
    ModelInfo,
    ModelType,
    Resnet50,
    Resnet50Fp32,
    Resnet50Int8,
    SSDMobileNet1,
    SSDMobileNet1Coco,
    TrtBertLargeFp32,
    TrtBertLargeInt8,
    TrtBertSmallFp32,
    TrtBertSmallInt8,
    TrtDummy,
    TrtEfficientLite4IR,
    TrtEfficientLite4ONNX,
    TrtGoogleNetIR,
    TrtGoogleNetONNX,
    TrtMatmulIR,
    TrtMatmulONNX,
    TrtResnet50Fp32,
    TrtResnet50Int8,
    TrtResnet50IR,
    TrtResnet50ONNX,
    TrtSSDMobileNet1Coco,
    TrtUnet3D,
    TrtVgg19IR,
    TrtVgg19ONNX,
    TrtYolo4Fp32,
    Unet3D,
    Vgg19,
    Yolo3Fp32,
    Yolo3TinyFp32,
    Yolo4Fp32,
)


class ModelsLibrary:
    def __init__(self):
        self.kpi_models = []

    def _get_instance_segmentation_security_shapes_no_full_auto(self):
        return [
            {"im_data": (1, 3, 800, 1344), "im_info": "auto"},
            {"im_data": "auto", "im_info": (1, 3)},
            {"im_data": (1, 3, 800, 1344), "im_info": (1, 3)},
        ]

    def _get_dummy_add_2_inputs_shapes_no_full_auto(self):
        return [
            {"input1": (1, 1000), "input2": "auto"},
            {"input1": "auto", "input2": (1, 1000)},
            {"input1": (1, 1000), "input2": (1, 1000)},
        ]

    def generate_model_shape_ids(self, shape):
        shape = re.sub(r"[\'{}()\s]", "", str(shape))
        shape = re.sub(r"[:,]", "_", shape)
        return f"shape__{shape}"

    def create_input_shapes_for_auto_reshape_tests(self, model_type: Type[ModelInfo]):
        shapes = ["auto"]
        if model_type:
            model = model_type()
            if len(model.input_shapes) > 1:
                for input_data in model.input_names:
                    shapes.append({input_data: "auto"})

            def _create_permutation(model: Type[ModelInfo], values):
                count = len(values) ** len(model.input_names)
                result = []
                for i in range(count):
                    tmp = {}
                    div = 1
                    for input_name in model.input_names:
                        shape = values[i // div % (len(values))](model, input_name)
                        tmp[input_name] = tuple(shape) if isinstance(shape, list) else shape
                        div *= len(values)
                    result.append(tmp)
                return result

            shapes += _create_permutation(model, [lambda m, arg: "auto", lambda m, arg: m.input_shapes[arg]])
        return shapes

    def get_many_models(self, device=None, limit=None):
        various_models = self.various_models[device]
        result = []
        if various_models:
            multiply = 4 if not limit else (int(limit / len(various_models)) + 1)
            models = list(chain(*[various_models for _ in range(multiply)]))
            for i, model_class in enumerate(models):
                model = model_class()
                model_path = model.get_model_path()
                old_model_name = model.name
                model.name = "{}-{}".format(model.name, i)
                model.model_path_on_host = os.path.join(models_path, old_model_name, str(model.version))
                model.base_path = model_path
                model.target_device = device
                result.append(model)
            if limit:
                result = result[:limit]
        return result

    def get_models_to_auto_download(self, device=None):
        models_to_autodownload = [ArgMax, Matmul, BasicLstm, Increment4d, InstanceSegmentationSecurity]
        models_to_autodownload += [OcrNetHrNetW48PaddleNative, OcrNetHrNetW48Paddle]
        models_to_autodownload += self.various_models[device]
        models_to_autodownload += self.models_used_in_example_client_apps
        models_to_autodownload += self.models_used_in_pipelines
        models_to_autodownload += self.stateful_models
        models_to_autodownload += self.resnet_wrong_models
        models_to_autodownload += [self.reshapeable_model_with_multiple_inputs[device][x] for x in ModelType]
        models_to_autodownload += [self.reshapeable_model[device][x] for x in ModelType]
        models_to_autodownload += self.models_used_for_benchmarking
        models_to_autodownload += self.various_models_for_add_another_model[device]
        models_to_autodownload += self.language_models
        return list(set(models_to_autodownload))

    @property
    def language_models(self):
        return [UniversalSentenceEncoder, Passthrough, Muse]

    @property
    def reshapeable_model(self):
        return defaultdict(
            dict,
            {
                TargetDevice.CPU: defaultdict(
                    lambda: None, {ModelType.IR: InstanceSegmentationSecurity, ModelType.ONNX: Matmul}
                ),
                TargetDevice.GPU: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Alexnet}
                ),
                TargetDevice.NPU: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Alexnet}
                ),
                TargetDevice.AUTO: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Alexnet}
                ),
                TargetDevice.HETERO: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Alexnet}
                ),
            },
        )

    @property
    def reshapeable_model_with_multiple_inputs(self):
        return defaultdict(
            dict,
            {
                TargetDevice.CPU: defaultdict(
                    lambda: None, {ModelType.IR: InstanceSegmentationSecurity, ModelType.ONNX: Matmul}
                ),
                TargetDevice.GPU: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Matmul}
                ),
                TargetDevice.NPU: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Matmul}
                ),
                TargetDevice.AUTO: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Matmul}
                ),
                TargetDevice.AUTO_CPU_GPU: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Matmul}
                ),
                TargetDevice.HETERO: defaultdict(
                    lambda: None, {ModelType.IR: DummyAdd2Inputs, ModelType.ONNX: Matmul}
                ),
            },
        )

    @property
    def scalar_inputs_models(self):
        return [ScalarDummy]

    @property
    def various_models(self):
        if self.kpi_models:
            return defaultdict(lambda: self.kpi_models)

        if all(elem == OvmsType.KUBERNETES for elem in ovms_types):
            return defaultdict(
                list,
                {
                    TargetDevice.CPU: self.various_models_kubernetes_cpu,
                    TargetDevice.GPU: self.various_models_kubernetes_gpu,
                },
            )
        else:
            return defaultdict(
                list,
                {
                    TargetDevice.CPU: self.various_models_cpu,
                    TargetDevice.GPU: self.various_models_gpu,
                    TargetDevice.NPU: self.various_models_npu,
                    TargetDevice.AUTO: self.various_models_auto,
                    TargetDevice.HETERO: self.various_models_hetero,
                },
            )

    @property
    def various_models_brain_included(self):
        result = self.various_models
        for device, model_types in result.items():
            if device in [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU, TargetDevice.AUTO]:
                result[device].append(Brain)
        return result

    @property
    def predict_models(self):
        result = defaultdict(list)
        for device, model_types in self.various_models.items():
            for model_type in model_types:
                model = model_type()
                got_dataset = model.inputs and all(
                    map(lambda x: x.get("dataset", None) is not None, model.inputs.values())
                )
                if got_dataset:
                    # Only models with proper dataset can be used in predict flow.
                    result[device].append(model_type)
        return result

    @property
    def predict_models_and_mediapipe(self):
        from tests.functional.constants.pipelines import SimpleModelMediaPipe
        result = self.predict_models
        for device, model_types in self.various_models.items():
            result[device].append(SimpleModelMediaPipe)
            if device in [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU, TargetDevice.AUTO]:
                result[device].append(Brain)
        return result

    @property
    def binary_io_models(self):
        models = [Resnet]
        return defaultdict(
            list,
            {
                TargetDevice.CPU: models,
                TargetDevice.GPU: models,
                TargetDevice.NPU: models,
                TargetDevice.AUTO: models,
                TargetDevice.HETERO: models,
            },
        )

    @property
    def custom_loader_models(self):
        result = defaultdict(list)
        excluded_models = (
            [OcrNetHrNetW48Paddle, OcrNetHrNetW48PaddleNative, DummySavedModel]
            + self.language_models
        )  # CVS-96757 CVS-108523
        for device, model_types in self.various_models.items():
            result[device] = [m for m in model_types if m not in excluded_models]
        return result

    @property
    def various_models_for_add_another_model(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: self.various_models_cpu,
                TargetDevice.GPU: self.various_models_gpu,
                TargetDevice.NPU: self.various_models_npu,
                TargetDevice.AUTO: self.various_models_auto,
                TargetDevice.HETERO: self.various_models_auto,
            },
        )

    def get_default_model(self, device=None):
        return self.various_models[device][0]

    @property
    def models_used_for_benchmarking(self):
        return [
            Dummy,
            Unet3D,
            Resnet50Int8,
            Resnet50Fp32,
            BertSmallInt8,
            BertSmallFp32,
            BertLargeInt8,
            BertLargeFp32,
            GoogleNet,
            Alexnet,
            Mnist,
            TfsResnet50Fp32,
            TfsResnet50Int8,
            TfsUnet3D,
            TrtResnet50Fp32,
            TrtResnet50Int8,
            TrtUnet3D,
            TrtDummy,
            TrtEfficientLite4IR,
            TrtGoogleNetIR,
            TrtResnet50IR,
            TrtVgg19IR,
            TrtMatmulIR,
            TrtEfficientLite4ONNX,
            TrtGoogleNetONNX,
            TrtResnet50ONNX,
            TrtVgg19ONNX,
            TrtMatmulONNX,
            TrtBertSmallInt8,
            TrtBertSmallFp32,
            TrtBertLargeInt8,
            TrtBertLargeFp32,
            TrtSSDMobileNet1Coco,
            TrtYolo4Fp32,
            Yolo3Fp32,
            Yolo3TinyFp32,
            Yolo4Fp32,
            MobileNet3Small,
            MobileNet3Large,
            SSDMobileNet1,
            SSDMobileNet1Coco,
            LSpeechV10,
            DLBertSmallUncasedInt8,
            DLDeeplabV3Int8,
            DLDensenet121Int8,
            DLEfficientnetD0Int8,
            DLGoogleNetV4Int8,
            DLMobileNetSSDInt8,
            DLMobileNetV2Int8,
            DLResnet50Int8New,
            DLResnet18Int8,
            DLSSDResnet34Int8,
            DLUnetCamvidInt8,
            DLYoloV3TinyInt8,
            DLYolo4Int8,
            DLBertSmallUncasedFp32,
            DLDeeplabV3Fp32,
            DLDensenet121Fp32,
            DLEfficientnetD0Fp32,
            DLGoogleNetV4Fp32,
            DLMobileNetSSDFp32,
            DLMobileNetV2Fp32,
            DLResnet18Fp32,
            DLResnet50Fp32,
            DLSSDResnet34Fp32,
            DLUnetCamvidFp32,
            DLYoloV3TinyFp32,
            DLYolo4Fp32,
        ]

    @property
    def models_used_in_pipelines(self):
        return [
            Resnet,
            GoogleNetV2Fp32,
            ArgMax,
            DummyIncrementDecrement,
            DummyIncrement,
            DummyAdd2Inputs,
            EastFp32,
            CrnnTf,
            Resnet50,
            Emotion,
            AgeGender,
            VehicleDetection,
            VehicleAttributesRecognition,
            FaceDetectionRetail,
        ]

    @property
    def models_used_in_example_client_apps(self):
        return [FaceDetection, Resnet50Binary]

    @property
    def stateful_models(self):
        return [RmLstm, AspireTdnn]

    @property
    def large_models(self):
        return [BertLargeFp32]

    @property
    def various_large_and_vision_language_models_on_commit(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [FacebookOpt125Int8, InternVL21BInt8],
                TargetDevice.GPU: [FacebookOpt125Int8, InternVL21BInt4],
                TargetDevice.NPU: [TinyLlama11BChatV10Int4SymCw, Phi35VisionInstructInt4SymCw],
            },
        )

    @property
    def various_large_and_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    # large
                    Llama27BChatHfInt8,
                    MetaLlama318BInstructInt8,
                    MetaLlama323BInstructInt8,
                    Mistral7BInstructv03Int8,
                    DeepSeekR1DistillQwen15BInt8,
                    # vision
                    MiniCPMV26Int8,
                    Qwen2VL7BInstructInt8,
                ],
                TargetDevice.GPU: [
                    Llama27BChatHfInt4,
                    MetaLlama318BInstructInt4,
                    MetaLlama323BInstructInt4,
                    Mistral7BInstructv03Int4,
                    # vision
                    MiniCPMV26Int4,
                    Qwen2VL7BInstructInt4,
                ],
                TargetDevice.NPU: [
                    Llama27BChatHfInt4SymCw,
                    MetaLlama318BInt4SymCw,
                    MetaLlama318BInstructInt4SymCw,
                    MetaLlama323BInstructInt4SymCw,
                    # vision
                    Phi35VisionInstructInt4SymCw,
                    Gemma34BItInt4SymCw,
                ],
            },
        )

    @property
    def various_legit_large_and_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    Llama27BChatHfInt8,
                    Qwen2VL7BInstructInt8,
                ],
                TargetDevice.GPU: [
                    Llama27BChatHfInt4,
                    Qwen2VL7BInstructInt4,
                ],
                TargetDevice.NPU: [
                    Llama27BChatHfInt4SymCw,
                    Phi35VisionInstructInt4SymCw,
                ],
            },
        )

    @property
    def various_dataset_large_and_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    Llama27BChatHfInt8,
                    MetaLlama323BInstructInt8,
                    MiniCPMV26Int8,
                    Qwen2VL7BInstructInt8,
                ],
                TargetDevice.GPU: [
                    Llama27BChatHfInt4,
                    MiniCPMV26Int4,
                    Qwen2VL7BInstructInt4,
                ],
                TargetDevice.NPU: [
                    Llama27BChatHfInt4SymCw,
                    MetaLlama323BInstructInt4SymCw,
                    Phi35VisionInstructInt4SymCw,
                ],
            },
        )

    @property
    def various_large_and_vision_language_models_stress_and_load(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [MetaLlama323BInstructInt8, Qwen2VL7BInstructInt8],
                TargetDevice.GPU: [Llama27BChatHfInt4, Qwen2VL7BInstructInt4],
                TargetDevice.NPU: [MetaLlama323BInstructInt4SymCw, Phi35VisionInstructInt4SymCw],
            },
        )

    @property
    def various_large_language_models_accuracy(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    # daily
                    MetaLlama323BInstructInt8,
                    MetaLlama318BInt8,
                    Mistral7BInstructv03Int8OvHf,
                    # weekly
                    Qwen257BInstructInt8,
                    SmolLM2135InstructGGUFFp16Hf,
                ] + extra_acc_models,
                TargetDevice.GPU: [
                    # daily
                    MetaLlama323BInstructInt4,
                    MetaLlama318BInt4,
                    Mistral7BInstructv03Int4SymCw,
                    # weekly
                    Qwen257BInstructInt4,
                    SmolLM2135InstructGGUFFp16Hf,
                ] + extra_acc_models,
                TargetDevice.NPU: [
                    # weekly
                    MetaLlama323BInstructInt4SymCw,
                    MetaLlama318BInt4SymCw,
                    Mistral7BInstructv03Int4CwOvHf,
                ] + extra_acc_models,
            },
        )

    @property
    def various_vision_language_models_accuracy(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    # daily
                    Gemma34bItInt8OvHf,
                    InternVL22BInt8OvHf,
                    Phi35VisionInstructInt8OvHf,
                    Qwen25VL7BInstructInt8OvHf,
                    Qwen3VL4BInstructInt8,
                    Qwen3VL8BInt4OvHf,
                    # weekly
                    Qwen2VL7BInstructInt8,
                    Llava157bHfInt8,
                ] + extra_acc_models,
                TargetDevice.GPU: [
                    # daily
                    Gemma34bItInt8OvHf,
                    InternVL22BInt8OvHf,
                    Phi35VisionInstructInt8OvHf,
                    Qwen25VL7BInstructInt8OvHf,
                    Qwen3VL4BInstructInt4,
                    Qwen3VL32BInstructInt4,
                    Qwen3VL8BInt4OvHf,
                    # weekly
                    Qwen2VL7BInstructInt4,
                    Llava157bHfInt4,
                ] + extra_acc_models,
                TargetDevice.NPU: [
                    # daily
                    Gemma34bItInt4CwOvHf,
                    Phi35VisionInstructInt4SymCw,
                    # weekly
                    Qwen2VL7BInstructInt4SymCw,
                ] + extra_acc_models,
            },
        )

    @property
    def various_simple_large_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [Llama27BChatHfInt8],
                TargetDevice.GPU: [Llama27BChatHfInt4],
                TargetDevice.NPU: [Llama27BChatHfInt4SymCw],
            },
        )

    @property
    def various_simple_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [InternVL21BInt8],
                TargetDevice.GPU: [InternVL21BInt4],
                TargetDevice.NPU: [Phi35VisionInstructInt4SymCw],
            },
        )

    @property
    def various_simple_large_and_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [Llama27BChatHfInt8, Qwen2VL7BInstructInt8] + extra_test_models,
                TargetDevice.GPU: [Llama27BChatHfInt4, Qwen2VL7BInstructInt4] + extra_test_models,
                TargetDevice.NPU: [Llama27BChatHfInt4SymCw, Phi35VisionInstructInt4SymCw] + extra_test_models,
            },
        )

    @property
    def various_large_and_vision_language_models_memory_check(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [MetaLlama323BInstructInt8, DeepSeekR1DistillQwen15BInt8, Qwen2VL7BInstructInt8],
                TargetDevice.GPU: [Llama27BChatHfInt4, Qwen2VL7BInstructInt4],
                TargetDevice.NPU: [MetaLlama323BInstructInt4SymCw, Phi35VisionInstructInt4SymCw],
            },
        )

    @property
    def various_large_language_models_long_prompt(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [MetaLlama323BInstructInt8, Qwen257BInstruct1MInt8],
                TargetDevice.GPU: [MetaLlama323BInstructInt4, Qwen257BInstruct1MInt4],
                TargetDevice.NPU: [MetaLlama323BInstructInt4SymCw, Qwen257BInstruct1MInt4SymCw],
            },
        )

    @property
    def single_large_language_models_tools_supported(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [Qwen38BInt8],
                TargetDevice.GPU: [Qwen38BInt4],
                TargetDevice.NPU: [Phi4MiniInstructInt4SymCw],
            },
        )

    @property
    def various_large_language_models_tools_supported(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [Hermes3Llama318BInt8, Phi4MiniInstructInt8],
                TargetDevice.GPU: [Hermes3Llama318BInt4, Phi4MiniInstructInt4],
                TargetDevice.NPU: [Hermes3Llama318BInt4SymCw, Phi4MiniInstructInt4SymCw],
            },
        )

    @property
    def various_large_language_models_bfcl_accuracy(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                                          # daily
                                          Qwen38BInt4OvHf,
                                          MetaLlama323BInstructInt8,
                                          Qwen3Coder30BA3BInt8OvHf,
                                          Qwen3Coder30BA3BInt4OvHf,
                                          GptOss20BInt4OvHf,
                                          GptOss20BInt8OvHf,
                                          Mistral7BInstructv03Int4,
                                          Qwen3VL4BInstructInt8,
                                          Qwen3VL8BInt4OvHf,
                                          # weekly
                                          Phi4MiniInstructInt8,
                                          Hermes3Llama318BInt8,
                                          DevstralSmall2507Int4,
                                      ] + extra_acc_models,
                TargetDevice.GPU: [
                                          # daily
                                          Qwen38BInt4OvHf,
                                          MetaLlama323BInstructInt4,
                                          Qwen3Coder30BA3BInt4OvHf,
                                          GptOss20BInt4OvHf,
                                          Mistral7BInstructv03Int4,
                                          Qwen3VL4BInstructInt4,
                                          Qwen3VL32BInstructInt4,
                                          Qwen3VL8BInt4OvHf,
                                          # weekly
                                          Phi4MiniInstructInt4,
                                          Hermes3Llama318BInt4,
                                          DevstralSmall2507Int4,
                                      ] + extra_acc_models,
                TargetDevice.NPU: [
                                          MetaLlama323BInstructInt4SymCw,
                                          Mistral7BInstructv03Int4SymCw,
                                      ] + extra_acc_models,
            },
        )

    @property
    def mini_language_model(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [FacebookOpt125Int8],
                TargetDevice.GPU: [FacebookOpt125Int8],
            },
        )

    @property
    def various_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    Qwen2VL7BInstructInt8,
                    MiniCPMV26Int8,
                    InternVL21BInt8,
                    Llava157bHfInt8,
                ],
                TargetDevice.GPU: [
                    Qwen2VL7BInstructInt4,
                    MiniCPMV26Int4,
                    InternVL21BInt4,
                    Llava157bHfInt4,
                ],
                TargetDevice.NPU: [
                    Phi35VisionInstructInt4SymCw,
                    Qwen2VL7BInstructInt4SymCw,
                ],
            },
        )

    @property
    def various_large_language_models_hf_simple(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    Qwen205BInt4OvHf,
                    Llama323BInstructQ4KMGGUFInt4Hf,
                    Qwen306BInstructGGUFInt8Hf,
                    SmolLM2135InstructGGUFFp16Hf,
                ],
                TargetDevice.GPU: [
                    Qwen205BInt4OvHf,
                    Llama323BInstructQ4KMGGUFInt4Hf,
                    Qwen306BInstructGGUFInt8Hf,
                    SmolLM2135InstructGGUFFp16Hf,
                ],
                TargetDevice.NPU: [
                    Qwen205BInt4OvHf,
                    Llama323BInstructQ4KMGGUFInt4Hf,
                    Qwen306BInstructGGUFInt8Hf,
                    SmolLM2135InstructGGUFFp16Hf,
                ],
            },
        )

    @property
    def all_large_language_and_vision_models_hf(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: models_llm_vlm_ov_hf,
                TargetDevice.GPU: models_llm_vlm_ov_hf,
                TargetDevice.NPU: models_llm_vlm_npu_ov_hf,
            },
        )

    @property
    def various_feature_extraction_models_on_commit(self):
        return [
            BAAIBgeLargeZhv15,
            Qwen3Embedding06B,
        ]

    @property
    def various_feature_extraction_models(self):
        return [
            AlibabaNLPGteLargeEnv15,
            BAAIBgeLargeEnv15,
            NomicEmbedTextv15,
            SentenceTransformersAllMpnetBaseV2,
            ThenlperGteSmall,
        ]

    @property
    def various_feature_extraction_models_accuracy(self):
        cpu_gpu_embeddings_models = [
            # daily
            BAAIBgeBaseEnv15Int8OvHf,
            Qwen3Embedding06BInt8OvHf,
            # weekly
            AlibabaNLPGteLargeEnv15,
            BAAIBgeLargeEnv15,
            BAAIBgeLargeZhv15,
            NomicEmbedTextv15,
            ThenlperGteSmall,
            SentenceTransformersAllMpnetBaseV2,
            SentenceTransformersAllMiniLML12V2,
            SentenceTransformersMultiQaMpnetBaseDotV1,
            SentenceTransformersAllDistilrobertaV1,
            MixedBreadAIDeepsetMxbaiEmbedLargeV1,
            IntfloatMultilingualE5Large,
            IntfloatMultilingualE5LargeInstruct,
        ]
        npu_models = [
            Qwen3Embedding06B,
            BAAIBgeLargeEnv15,
            BAAIBgeLargeZhv15,
            ThenlperGteSmall,
            SentenceTransformersAllMpnetBaseV2,
        ]
        return defaultdict(
            list,
            {
                TargetDevice.CPU: cpu_gpu_embeddings_models,
                TargetDevice.GPU: cpu_gpu_embeddings_models,
                TargetDevice.NPU: npu_models,
            },
        )

    @property
    def various_audio_models_accuracy(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [WhisperLargeV3TurboFp16],
                TargetDevice.GPU: [WhisperLargeV3TurboFp16],
                TargetDevice.NPU: [WhisperLargeV3TurboInt8],
            },
        )

    @property
    def various_feature_extraction_models_hf(self):
        return [
            BAAIBgeBaseEnv15Int8OvHf,
            BAAIBgeBaseEnv15Fp16OvHf,
            Qwen3Embedding06BInt8OvHf,
            Qwen3Embedding06BFp16OvHf,
        ]

    @property
    def various_rerank_models_on_commit(self):
        return [
            BAAIRerankerLarge,
            BAAIRerankerV2M3,
            BAAIRerankerBase,
        ]

    @property
    def various_rerank_models(self):
        return [
            Qwen3Reranker06BSeqCls,
            CrossEncoderMsmarcoMiniLML6EnDeV1,
        ]

    @property
    def various_rerank_models_hf(self):
        return [
            BAAIRBgeRerankerBaseInt8OvHf,
            BAAIRBgeRerankerBaseFp16OvHf,
            Qwen3Reranker06BSeqClsFp16OvHf,
        ]

    @property
    def various_image_generation_models_on_commit(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [StableDiffusionv15Int8OvHf],
                TargetDevice.GPU: [StableDiffusionv15Int8OvHf],
                TargetDevice.NPU: [StableDiffusionv15Int8OvHf],
            },
        )

    @property
    def various_image_generation_models_stress(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [DreamlikeAnime10Int8] + extra_test_models,
                TargetDevice.GPU: [DreamlikeAnime10Int4] + extra_test_models,
                TargetDevice.NPU: [DreamlikeAnime10Int4] + extra_test_models,
            },
        )

    @property
    def various_image_generation_models(self):
        # check all image generation models architectures (Flux, SD, SD3, SDXL)
        # architecture can be checked in model_index.json file
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    Flux1SchnellInt8OvHf,    # flux architecture
                    DreamlikeAnime10Int8, LCMDreamshaperv7Int8OvHf,  # SD architecture
                    StableDiffusion35LargeTurboInt8,    # SD3 architecture
                    StableDiffusionXlBase10Int8,    # SDXL architecture
                ],
                TargetDevice.GPU: [
                    Flux1SchnellInt4OvHf,
                    DreamlikeAnime10Int4, LCMDreamshaperv7Int8OvHf,
                    StableDiffusion35LargeTurboInt4,
                    StableDiffusionXlBase10Int4,
                ],
                TargetDevice.NPU: [
                    # Flux model not yet supported by NPU: EISW-144732
                    DreamlikeAnime10Int4SymCw, LCMDreamshaperv7Int8OvHf,
                    StableDiffusion35LargeTurboInt4SymCw,
                    StableDiffusionXlBase10Int4SymCw,
                ],
            },
        )

    @property
    def various_image_inpainting_models_on_commit(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [DreamlikeDiffusion10InpaintingInt8],
                TargetDevice.GPU: [DreamlikeDiffusion10InpaintingInt4],
                TargetDevice.NPU: [DreamlikeDiffusion10InpaintingInt4SymCw],
            },
        )

    @property
    def various_image_inpainting_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [
                    StableDiffusionXl10Inpainting01Int8,
                    StableDiffusionInpaintingInt8,
                    DreamlikeAnime10Int8,
                    StableDiffusionv15Int8OvHf,
                    Dreamshaper8InpaintingInt8OvHf,
                    Flux1SchnellInt8OvHf,
                ],
                TargetDevice.GPU: [
                    StableDiffusionXl10Inpainting01Int4,
                    StableDiffusionInpaintingInt4,
                    DreamlikeAnime10Int4,
                    StableDiffusionv15Int8OvHf,
                    Dreamshaper8InpaintingInt8OvHf,
                    Flux1SchnellInt4OvHf,
                ],
                TargetDevice.NPU: [
                    StableDiffusionXl10Inpainting01Int8,
                    StableDiffusionInpaintingInt8,
                    DreamlikeAnime10Int8,
                    StableDiffusionv15Int8OvHf,
                    Dreamshaper8InpaintingInt8OvHf,
                ],
            },
        )

    @property
    def various_audio_models_tts_on_commit(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [MicrosoftSpeech5TtsInt8],
                TargetDevice.GPU: [MicrosoftSpeech5TtsInt8],
            },
        )

    @property
    def various_audio_models_tts(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [MicrosoftSpeech5TtsFp16, MicrosoftSpeech5TtsInt8,
                                       Sandiago21Speech5TtsSpanishFp16],
                TargetDevice.GPU: [MicrosoftSpeech5TtsFp16, MicrosoftSpeech5TtsInt8,
                                       Sandiago21Speech5TtsSpanishFp16],
            },
        )

    @property
    def various_audio_models_tts_stress_and_load(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [MicrosoftSpeech5TtsInt8, Sandiago21Speech5TtsSpanishFp16],
                TargetDevice.GPU: [MicrosoftSpeech5TtsInt8, Sandiago21Speech5TtsSpanishFp16],
            },
        )

    @property
    def various_audio_models_asr_on_commit(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [WhisperSmallInt8],
                TargetDevice.GPU: [WhisperSmallInt8],
                TargetDevice.NPU: [WhisperSmallInt8],
            },
        )

    @property
    def various_audio_models_asr(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [WhisperSmallInt8, WhisperLargeV3TurboInt8, WhisperLargeV3Int8,
                                       WhisperLargeV3Int8OvHf, DistilWhisperLargeV3Int8OvHf],
                TargetDevice.GPU: [WhisperSmallInt8, WhisperLargeV3TurboInt4, WhisperLargeV3Int4,
                                       WhisperLargeV3Int4OvHf, DistilWhisperLargeV3Int8OvHf],
                TargetDevice.NPU: [WhisperLargeV3Int4, WhisperLargeV3TurboInt4],
            },
        )

    @property
    def various_audio_models_asr_weekly(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [WhisperSmallFp16, WhisperLargeV3TurboFp16, WhisperLargeV3Fp16,
                                       WhisperLargeV3TurboInt4, WhisperLargeV3Int4,
                                       WhisperLargeV3Int4OvHf, WhisperLargeV3Fp16OvHf],
                TargetDevice.GPU: [WhisperSmallFp16, WhisperLargeV3TurboFp16, WhisperLargeV3Fp16,
                                       WhisperLargeV3TurboInt8, WhisperLargeV3Int8,
                                       WhisperLargeV3Int8OvHf, WhisperLargeV3Fp16OvHf],
                TargetDevice.NPU: [WhisperSmallFp16],
            },
        )

    @property
    def various_audio_models_asr_regression(self):
        asr = self.various_audio_models_asr
        asr_weekly = self.various_audio_models_asr_weekly
        return defaultdict(list, {k: asr[k] + asr_weekly[k] for k in set(asr) | set(asr_weekly)})

    @property
    def various_audio_models_asr_stress_and_load(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [WhisperLargeV3Int4, WhisperLargeV3TurboInt4],
                TargetDevice.GPU: [WhisperLargeV3Int4, WhisperLargeV3TurboInt4],
                TargetDevice.NPU: [WhisperLargeV3Int4, WhisperLargeV3TurboInt4],
            },
        )

    @property
    def various_models_cpu(self):
        various_models_cpu = [
            Resnet,
            SsdliteMobilenetV2,
            InceptionResnetV2,
            InstanceSegmentationSecurity,
            # Enable only specific tests for OcrNetHrNetW48Paddle* models.
            # OcrNetHrNetW48Paddle,
            # OcrNetHrNetW48PaddleNative,
        ]
        if language_models_enabled:
            various_models_cpu.extend([DummySavedModel, UniversalSentenceEncoder, Passthrough])
        return various_models_cpu

    @property
    def various_models_mediapipe(self):
        default_model_list = [Resnet, SsdliteMobilenetV2, InceptionResnetV2]
        cpu_model_list = default_model_list + [InstanceSegmentationSecurity]
        various_models_mediapipe = defaultdict(lambda: default_model_list, {TargetDevice.CPU: cpu_model_list})
        return various_models_mediapipe

    @property
    def various_models_single_in_out_mediapipe(self):
        various_models_single_in_out_mediapipe = [Dummy, Resnet, SsdliteMobilenetV2, InceptionResnetV2]
        return various_models_single_in_out_mediapipe

    @property
    def various_models_and_simple_model_mediapipe_brain_included(self):
        various_models_and_simple_model_mediapipe = self.various_models_and_simple_model_mediapipe
        for device, model_types in various_models_and_simple_model_mediapipe.items():
            if device in [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU, TargetDevice.AUTO]:
                various_models_and_simple_model_mediapipe[device].append(Brain)
        return various_models_and_simple_model_mediapipe

    @property
    def various_models_and_simple_model_mediapipe(self):
        from tests.functional.constants.pipelines import SimpleModelMediaPipe
        various_models_mediapipe = self.various_models_mediapipe
        for key in various_models_mediapipe.keys():
            various_models_mediapipe[key].extend([SimpleModelMediaPipe])
        return various_models_mediapipe

    @property
    def various_ov_models(self):
        various_ov_models = [Resnet, SsdliteMobilenetV2, InceptionResnetV2, Brain, OcrNetHrNetW48PaddleNative]
        return various_ov_models

    @property
    def various_ov_models_and_dummy(self):
        various_ov_models = self.various_ov_models
        various_ov_models.extend([DummyIncrement])
        return various_ov_models

    @property
    def various_models_gpu(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2]

    @property
    def various_models_npu(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2]

    @property
    def various_models_myriad(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2, Resnet50]

    @property
    def various_models_hddl(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2]

    @property
    def various_models_auto(self):
        return [Resnet, InceptionResnetV2, SsdliteMobilenetV2]

    @property
    def various_models_hetero(self):
        return [Resnet, InceptionResnetV2, SsdliteMobilenetV2]

    @property
    def other_models(self):
        return [UnsupportedModel]

    @property
    def various_models_kubernetes_cpu(self):
        return [
            Resnet,
            InceptionResnetV2,
            InstanceSegmentationSecurity,
            OcrNetHrNetW48Paddle,
            OcrNetHrNetW48PaddleNative,
        ]

    @property
    def various_models_kubernetes_gpu(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2]

    @property
    def onnx_various_model_cpu(self):
        return [Alexnet, Caffenet, Densenet121, EfficientLite4, GoogleNet, Mnist, RcnnIlsvrc13, Resnet50, Vgg19]

    @staticmethod
    def get_resnet_models_with_wrong_input_types(base_model=None):
        wrong_type_list = [int, np.float64]
        result = []
        for wrong_type in wrong_type_list:
            model = base_model
            model.inputs["map/TensorArrayStack/TensorArrayGatherV3"]["dtype"] = wrong_type
            result.append(model)
        return result

    @property
    def resnet_wrong_models(self):
        return [NoModel, Yamnet, WrongExtensionModel]

    @property
    def models_with_wrong_xml(self):
        return [WrongXmlEmpty, WrongXmlWithout, WrongXmlSubdirectory]

    @property
    def models_with_encoded_names(self):
        return [ResnetModelNameWithSlash, ResnetModelNameWithWhitespace]

    @property
    def model_shapes_no_full_auto(self):
        default = self._get_dummy_add_2_inputs_shapes_no_full_auto()
        model_shapes_dict = defaultdict(lambda: default)
        for target_device, models in self.reshapeable_model.items():
            if models[ModelType.IR] == InstanceSegmentationSecurity:
                model_shapes_dict[target_device] = self._get_instance_segmentation_security_shapes_no_full_auto()
        return model_shapes_dict

    def get_model_with_wrong_path_and_expected_ovms_msg(self):
        result = []
        not_existing_model_path = NotExistingModelPath()
        result.append((not_existing_model_path, OvmsMessages.INVALID_MODEL_PATH))

        for model in self.resnet_wrong_models:
            if model is Yamnet:
                result.append(
                    (model(), OvmsMessages.ERROR_LOADING_MODEL_INTERNAL_SERVER_ERROR.format(model.name, model.version))
                )
            else:
                result.append((model(), OvmsMessages.NOT_FOUND_MODEL_IN_PATH))
        return result

    def get_model_with_wrong_path_and_expected_ovms_msg_ids(self):
        # [NoModel, NotExistingModelPath, Yamnet, WrongExtensionModel]
        return ("non-existent_path", "empty_directory", "model_non_supported", "extension_changed_to_wrong_one")

    def create_model(self, name):
        children = get_children_from_module(ModelInfo, ovms.constants.models)  # [(name, class_def), ...]
        # we don't need 'name', we require only 'class_def' (without possible duplicates)
        children = list(set(map(lambda x: x[1], children)))
        result = list(filter(lambda x: x.name == name, children))
        assert result is not None and len(result) == 1, f"Expected single result, but got {result}"
        model = result[0]()  # instantiate
        return model

    @property
    def models_with_auto_batch_size(self):
        return defaultdict(list, {TargetDevice.CPU: [Resnet, InceptionResnetV2]})

    @property
    def models_with_dynamic_shape_support(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [Resnet, InceptionResnetV2],
            },
        )

    def generate_ids_for_iteration_info(self, iteration_info):
        num_of_models_str = "single_model"
        num_of_models = len(iteration_info)
        if num_of_models > 1:
            num_of_models_str = "many_models"

        enable_cache_str = "cache_disabled"
        use_custom_loader_str = "without_custom_loader"
        for model, enable_cache, use_custom_loader in iteration_info:
            if enable_cache is None:
                enable_cache_str = "cache_default"
                break
            elif enable_cache:
                enable_cache_str = "cache_enabled"
                break

        for model, enable_cache, use_custom_loader in iteration_info:
            if use_custom_loader:
                use_custom_loader_str = "with_custom_loader"
                break

        return f"{num_of_models_str}_{enable_cache_str}_{use_custom_loader_str}"


ModelsLib = ModelsLibrary()
