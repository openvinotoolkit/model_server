# Mediapipe calculators {#ovms_docs_mediapipe_calculators}

## Introduction

"Each calculator is a node of a graph. The bulk of graph execution happens inside its calculators. OpenVINO Model Server(OVMS) has its own calculators but can also use newly developed calculators or reuse the existing calculators defined in the original mediapipe repository."

For more details about the calculator concept you can visit mediapipe [Calculators Concept Page](https://developers.google.com/mediapipe/framework/framework_concepts/calculators)

### OVMSInferenceAdapter

Is an implementation of [Intel Model API](https://github.com/openvinotoolkit/model_api) Adapter [interface](https://github.com/openvinotoolkit/model_api/blob/master/model_api/cpp/adapters/include/adapters/inference_adapter.h) that executes inference with OVMS [C-API](https://github.com/openvinotoolkit/model_server/blob/main/docs/model_server_c_api.md).

### OpenVINOModelServerSessionCalculator

This [calculator](https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/calculators/ovms/openvinomodelserversessioncalculator.cc) is creating OVMS Adapter to declare what model/[DAG](https://github.com/openvinotoolkit/model_server/blob/main/docs/dag_scheduler.md) should be used in inference. It has mandatory field `servable_name` and optional `servable_version`. In case of missing `servable_version` OVMS will use default version for targeted servable. Another optional field is 'server_config' which is file path to OVMS configuration file. In case of using graph inside OVMS this field is ignored. In case of running inside any application using Mediapipe, the first calculator will trigger server start through OVMS [C-API](https://github.com/openvinotoolkit/model_server/blob/main/docs/model_server_c_api.md)

### OpenVINOInferenceCalculator

This [calculator](https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/calculators/ovms/openvinoinferencecalculator.cc) is using OVMS Adapter received as `input_side_packet` to execute inference with OVMS. It has optional options fields `tag_to_input_tensor_names` and `tag_to_output_tensor_names` that can serve as Mediapipe stream names mapping to servable (Model/DAG) inputs and/or outputs. Options `input_order_list` and `output_order_list` can be used in conjuction with packet types using `std::vector<T>` to transform input/output maps to desired order in vector of tensors. Example of usage can be found [here](https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt).

Accepted packet types and tags are listed below:

|pbtxt line|input/output|tag|packet type|stream name|
|:---|:---|:---|:---|:---|
|input_stream: "a"|input|none|ov::Tensor|a|
|output_stream: "OVTENSOR:b"|output|OVTENSOR|ov::Tensor|b|
|output_stream: "OVTENSORS:b"|output|OVTENSORS|std::vector<ov::Tensor>|b|
|output_stream: "TENSOR:b"|output|TENSOR|mediapipe::Tensor|b|
|input_stream: "TENSORS:b"|input|TENSORS|std::vector<mediapipe::Tensor>|b|

In case of missing tag calculator assumes that the packet type is `ov::Tensor'.
