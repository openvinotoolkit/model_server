---
id: ovms_docs_binary_input
sidebar_position: 2
---

# Support for Binary Encoded Image Input Data


For images, to reduce data size and lower bandwidth usage you can send them in binary-encoded instead of array-like format. How you can do it, depends on the kind of servable.

**Single Models and DAG Pipelines**:

While OpenVINO models don't have the ability to process images directly in their binary format, the model server can accept them and convert
automatically from JPEG/PNG to OpenVINO friendly format using built-in [OpenCV](https://opencv.org/) library. To take advantage of this feature, there are two requirements:
   1. Model input, that receives binary encoded image, must have a proper shape and layout. Learn more about this requirement in [input shape and layout considerations](binary-input-layout-and-shape.md) document.
   2. Inference request sent to the server must have certain properties. These properties are different depending on the interface (gRPC or REST). Learn more:
      - [KServe API](binary-input-kfs.md)

With KServe API, you can also send raw data with or without image encoding via REST API. The guide linked above explains how to work with both regular data in binary format as well as JPEG/PNG encoded images.

**MediaPipe Graphs**:

When serving MediaPipe Graph it is possible to configure it to accept binary encoded images. You can either create your own calculator that would implement image decoding and use it in the graph or use `PythonExecutorCalculator` and implement decoding in Python [execute function](../reference/python-support-reference.md#ovmspythonmodel-class).
