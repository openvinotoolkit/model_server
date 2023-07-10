# Dynamic Input Parameters {#ovms_docs_dynamic_input}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_dynamic_bs_demultiplexer
   ovms_docs_dynamic_bs_auto_reload
   ovms_docs_dynamic_shape_auto_reload
   ovms_docs_dynamic_shape_custom_node
   ovms_docs_dynamic_shape_binary_inputs
   ovms_docs_dynamic_shape_dynamic_model


@endsphinxdirective

Models served by the OpenVINO Model Server can be configured to accept data with different batch sizes and in different shapes.
There are multiple ways of enabling dynamic inputs for the model:

- [dynamic batch size with a demultiplexer](./dynamic_bs_demultiplexer.md) - create a simple pipeline that splits data of any batch size and performs inference on each element in the batch separately. Consider using this option if incoming requests will be containing various batch size. This option does not need to reload underlying model, therefore there is no model reloading impact on the performance.

- [dynamic batch size with automatic model reloading](./dynamic_bs_auto_reload.md) - configure the Model Server to reload the model each time it receives a request with a batch size other than what is currently set. Consider using this option when request batch size may change, but usually stays the same. Each request with varying batch size will impact the performance due to model reloading.

- [dynamic shape with automatic model reloading](./dynamic_shape_auto_reload.md) - configure the Model Server to reload the model each time the model receives a request with data in shape other than what is currently set. Consider using this option when request shape may change, but usually stays the same. Each request with varying shape will impact the performance due to model reloading.

- [dynamic input shape with a custom node](./dynamic_shape_custom_node.md) - create a simple pipeline by pairing your model with a custom node that performs data preprocessing and provides the model with data in an acceptable shape. Consider this option if you want to fit the image into model shape by performing image resize operation before inference. This may affect accuracy.

- [dynamic input shape with binary input format](./dynamic_shape_binary_inputs.md) - send data in binary format (JPEG or PNG encoded), so the Model Server will adjust the input during data decoding. Consider this option in case of slower networks to minimize amount of data transferred over the network and fit image size to the size accepted by endpoint.

- [dynamic input shape with dynamic IR/ONNX model](./dynamic_shape_dynamic_model.md) - leverage OpenVINO native dynamic shape feature to send data with arbitrary shape. Consider using this option if model accepts dynamic dimensions.
