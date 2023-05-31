# Mediapipe calculators {#ovms_docs_mediapipe_calculators}

## Introduction

"Each calculator is a node of a graph. The bulk of graph execution happens inside its calculators. OpenVINO Model Server(OVMS) has its own calculators but can also use newly developed calculators or reuse the existing calculators defined in the original mediapipe repository."

For more details about the calculator concept you can visit mediapipe [Calculators Concept Page](https://developers.google.com/mediapipe/framework/framework_concepts/calculators)

### OVMS ADAPTER

Is an implementation of [Intel Model API](https://github.com/openvinotoolkit/model_api) Adapter [interface](https://github.com/openvinotoolkit/model_api/blob/master/model_api/cpp/adapters/include/adapters/inference_adapter.h) that executes inference with OVMS [C-API](https://github.com/openvinotoolkit/model_server/blob/develop/docs/model_server_c_api.md).

### OVMS SESSION CALCULATOR

This [calculator](https://github.com/openvinotoolkit/model_server/blob/develop/src/mediapipe_calculators/modelapiovmsinferencecalculator.cc) is creating OVMS Adapter to declare what model/[DAG](https://github.com/openvinotoolkit/model_server/blob/develop/docs/dag_scheduler.md) should be used in inference. It has mandatory field `servable_name` and optional `servable_version`. In case of missing `servable_version` OVMS will use default version for targeted servable.

### OVMS INFERENCE CALCULATOR

This [calculator](https://github.com/openvinotoolkit/model_server/blob/develop/src/mediapipe_calculators/modelapiovmssessioncalculator.cc) is using OVMS Adapter received as `input_side_packet` to execute inference with OVMS. It has optional options fields `tag_to_input_tensor_names` and `tag_to_output_tensor_names` that can serve as Mediapipe packet names mapping to servable (Model/DAG) inputs and/or outputs. It accepts `ov::Tensor` as input and output packet types.

