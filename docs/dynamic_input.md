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

- Handling dynamic batch size with a demuliplexer

- Handling dynamic batch size with automatic model reloading

- Handling dynamic shape with automatic model reloading

- Handling dynamic input shape with a custom node

- Handling dynamic input shape with binary input format

Each of them has been described in this section.