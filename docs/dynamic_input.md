# Dynamic Input Parameters {#ovms_docs_dynamic_input}

```{toctree}
---
maxdepth: 1
hidden:
---

ovms_docs_dynamic_shape_dynamic_model
ovms_docs_dynamic_shape_binary_inputs
ovms_docs_dynamic_bs_auto_reload
ovms_docs_dynamic_shape_auto_reload
ovms_docs_dynamic_bs_demultiplexer
ovms_docs_dynamic_shape_custom_node
```

OpenVINO Model Server servables can be configured to accept data with different batch sizes and in different shapes.
There are multiple ways of enabling dynamic inputs depending on the kind of servable.

**Single Models**:

- [dynamic input shape with dynamic IR/ONNX model](./dynamic_shape_dynamic_model.md) - leverage OpenVINO native dynamic shape feature to send data with arbitrary shape. Consider using this option if model accepts dynamic dimensions.

- [dynamic input shape with binary input format](./dynamic_shape_binary_inputs.md) - send data in binary format (JPEG or PNG encoded), so the Model Server will adjust the input during data decoding. Consider this option in case of slower networks to minimize amount of data transferred over the network and fit image size to the size accepted by endpoint.

- [**DEPRECATED**] [dynamic batch size with automatic model reloading](./dynamic_bs_auto_reload.md) - configure the Model Server to reload the model each time it receives a request with a batch size other than what is currently set. Consider using this option when request batch size may change, but usually stays the same. Each request with varying batch size will impact the performance due to model reloading.

- [**DEPRECATED**] [dynamic shape with automatic model reloading](./dynamic_shape_auto_reload.md) - configure the Model Server to reload the model each time the model receives a request with data in shape other than what is currently set. Consider using this option when request shape may change, but usually stays the same. Each request with varying shape will impact the performance due to model reloading.

**DAG Pipelines**:

- [dynamic batch size with a demultiplexer](./dynamic_bs_demultiplexer.md) - create a simple pipeline that splits data of any batch size and performs inference on each element in the batch separately. Consider using this option if incoming requests will be containing various batch size. This option does not need to reload underlying model, therefore there is no model reloading impact on the performance.

- [dynamic input shape with a custom node](./dynamic_shape_custom_node.md) - create a simple pipeline by pairing your model with a custom node that performs data preprocessing and provides the model with data in an acceptable shape. Consider this option if you want to fit the image into model shape by performing image resize operation before inference. This may affect accuracy.

**MediaPipe Graphs**:

OpenVINO Model Server accepts several data types that can be handled on [MediaPipe graph](mediapipe.md) input. Whether the input is dynamic or not depends on what happens next in the graph. There are 4 situations when input to the graph can be dynamic:

- Next node in the graph uses `OpenVINOInferenceCalculator` that runs inference on a model that accepts dynamic inputs. Such node expects input stream with a tag starting with `OVTENSOR` prefix.

- Next node in the graph uses a calculator that handles input in [MediaPipe ImageFrame](https://developers.google.com/mediapipe/api/solutions/python/mp/ImageFrame) format. Model Server converts data from the KServe request to MediaPipe ImageFrame for input streams which tags start with `IMAGE` prefix.

- Next node in the graph uses a calculator that can decode raw KServe request. In such case dynamic input handling must be implemented as part of the calculator logic since model server passes the request to the calculator as-is. Such node expects input stream with a tag starting with `REQUEST` prefix.

- Next node in the graph uses `PythonExecutorCalculator`. In such case data in the KServe request will be available to the user as input argument of their Python [execute function](https://github.com/openvinotoolkit/model_server/blob/releases/2025/3/docs/python_support/reference.md#ovmspythonmodel-class). Such node expects input stream with a tag starting with `OVMS_PY_TENSOR` prefix.

