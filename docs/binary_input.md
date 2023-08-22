# Support for Binary Encoded Image Input Data {#ovms_docs_binary_input}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_binary_input_layout_and_shape
   ovms_docs_binary_input_tfs
   ovms_docs_binary_input_kfs
   ovms_docs_demo_tensorflow_conversion

@endsphinxdirective

While OpenVINO models don't have the ability to process images directly in their binary format, the model server can accept them and convert
automatically from JPEG/PNG to OpenVINO friendly format using built-in [OpenCV](https://opencv.org/) library. To take advantage of this feature, there are two requirements:
   1. Model input, that receives binary encoded image, must have a proper shape and layout. Learn more about this requirement in [input shape and layout considerations](./binary_input_layout_and_shape.md) document.
   2. Inference request sent to the server must have certain properties. These properties are different depending on the API (KServe or TensorFlow Serving) and interface (gRPC or REST). Choose the API you are using and learn more:
      - [TensorFlow Serving API](./binary_input_tfs.md)
      - [KServe API](./binary_input_kfs.md)

It's worth noting that with KServe API, you can also send raw data with or without image encoding via REST API. This makes KServe REST API more performant choice comparing to json format in TFS API. The guide linked above explains how to work with both regular data in binary format as well as JPEG/PNG encoded images. 