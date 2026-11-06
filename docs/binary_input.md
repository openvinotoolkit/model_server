# Support for Binary Encoded Image Input Data {#ovms_docs_binary_input}

```{toctree}
---
maxdepth: 1
hidden:
---

ovms_docs_binary_input_layout_and_shape
ovms_docs_binary_input_kfs
ovms_docs_demo_tensorflow_conversion
```

For images, to reduce data size and lower bandwidth usage you can send them in binary-encoded instead of array-like format. How you can do it, depends on the kind of servable.

**Single Models and DAG Pipelines**:

While OpenVINO models don't have the ability to process images directly in their binary format, the model server can accept them and convert
automatically from JPEG/PNG to OpenVINO tensors. OpenCV decoding is preferred when its optional runtime plugin and libraries are available; otherwise OVMS uses stb_image. Resizing and precision conversion use OpenVINO preprocessing in either case. To take advantage of this feature, there are two requirements:
   1. Model input, that receives binary encoded image, must have a proper shape and layout. Learn more about this requirement in [input shape and layout considerations](./binary_input_layout_and_shape.md) document.
   2. Inference request sent to the server must have certain properties. These properties are different depending on the interface (gRPC or REST). Learn more:
      - [KServe API](./binary_input_kfs.md)

With KServe API, you can also send raw data with or without image encoding via REST API. The guide linked above explains how to work with both regular data in binary format as well as JPEG/PNG encoded images.

### Optional OpenCV decoding

OpenCV is not required to start OVMS or serve ordinary numeric tensors. The image decoder is selected once, on the first encoded-image request:

- If the OpenCV decoder plugin can be loaded, requests use OpenCV. A decoding failure is returned to the client; OVMS does not silently retry malformed images with another decoder.
- Otherwise requests use stb_image. JPEG and PNG are supported, including grayscale, alpha, and 16-bit PNG. KServe retains BGR/BGRA channel order, unlike the RGB tensor used by the LLM image loader.
- Formats supported only by OpenCV, such as WebP, JPEG 2000, and PFM, require the plugin and the corresponding OpenCV codecs. Without them, decoding fails with an error that explicitly identifies the missing or incompatible OpenCV backend and suggests installing it.

The Linux plugin is `libovms_opencv_image_decoder.so`; the Windows plugin is `ovms_opencv_image_decoder.dll`. Install it alongside the OVMS library, or in the release package's `lib` directory, with its OpenCV dependencies discoverable by the OS loader. Restart OVMS after installing or replacing a decoder plugin. Plugins must match the OVMS/OpenVINO build.

OVMS validates image dimensions and the per-request pixel budget before decoding, regardless of backend. Allowing unestimatable formats does not add codecs to stb_image; see [image decoding limits](./parameters.md).

OpenVINO preprocessing preserves conversion-before-resize ordering, batch validation, and resolution selection. Compiled preprocessing graphs are cached with a bounded number of entries; inference requests and output buffers are independent for concurrent requests. This changes integer resize rounding compared with the previous OpenCV implementation: OpenVINO rounds float interpolation results, whereas OpenCV uses a fixed-point path for u8. Compatibility tests cover a maximum absolute difference of one intensity level on representative u8/u16 resize cases. Different JPEG decoders can introduce additional differences; do not rely on bit-identical output across backends.

**MediaPipe Graphs**:

When serving MediaPipe Graph it is possible to configure it to accept binary encoded images. You can either create your own calculator that would implement image decoding and use it in the graph or use `PythonExecutorCalculator` and implement decoding in Python [execute function](./python_support/reference.md#ovmspythonmodel-class).
