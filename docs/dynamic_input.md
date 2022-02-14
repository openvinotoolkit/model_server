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


@endsphinxdirective

Models served by the OpenVINO Model Server can be configured to accept data with different batch sizes and in different shapes.
There are multiple ways of enabling dynamic inputs for the model:

- [dynamic batch size with a demuliplexer](./dynamic_bs_demultiplexer.md) - create a simple pipeline that splits data of any batch size and performs inference on each element in the batch separately.

- [dynamic batch size with automatic model reloading](./dynamic_bs_auto_reload.md) - configure the Model Server to reload the model each time it receives a request with a batch size other than what is currently set.

- [dynamic shape with automatic model reloading](./dynamic_shape_auto_reload.md) - configure the Model Server to reload the model each time the model receives a request with data in shape other than what is currently set.

- [dynamic input shape with a custom node](./dynamic_shape_custom_node.md) - create a simple pipeline by pairing your model with a custom node that performs data preprocessing and provides the model with data in an acceptable shape.

- [dynamic input shape with binary input format](./dynamic_shape_binary_inputs.md) - send data in binary format (JPEG or PNG encoded), so the Model Server will adjust the input during data decoding. 
