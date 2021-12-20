# Dynamic input parameters {#ovms_docs_dynamic_input}

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
There are mutiple ways of enabling dynamic inputs for the model, namely:

- [Handling dynamic batch size with a demuliplexer](./dynamic_bs_demultiplexer.md) - create a simple pipeline that will split data of any batch size and perform inference on each element in the batch separately.

- [Handling dynamic batch size with automatic model reloading](./dynamic_bs_auto_reload.md) - configure model server to reload the model every time it receives a request with batch size other than currently set.

- [Handling dynamic shape with automatic model reloading](./dynamic_shape_auto_reload.md) - configure model server to reload the model every time model receives a request with data in shape other than currently set.

- [Handling dynamic input shape with a custom node](./dynamic_shape_custom_node.md) - create a simple pipeline by pairing your model with a custom node that will perform data preprocessing and provide your model with data in acceptable shape.

- [Handling dynamic input shape with binary input format](./dynamic_shape_binary_inputs.md) - send data in binary format (JPEG or PNG encoded), so model server will adjust the input on data decoding. 
