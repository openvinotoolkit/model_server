# GStreamer VA Display Runbook

The `va_display` configuration establishes one server-owned VA display before
GPU models are loaded. The configured `target_device` and `render_node` are an
explicit pair; do not infer a DRM node from `GPU.0` or `GPU.1`.

```json
"va_display": {
  "target_device": "GPU.0",
  "render_node": "/dev/dri/renderD128"
}
```

Every GPU model using this path must set `target_device` to the same value.
The server initializes this display before loading models, injects it into
GStreamer, and releases it only after servables have shut down. There is no
implicit render-node fallback. The current implementation supports one display
for one GPU target per OVMS process.

## GPU Build Container

Create the development container with DRM access and the render-node group:

```bash
docker run -dit --name ovms_gst_va2 --entrypoint bash \
  --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -v "$(pwd)":/ovms ovms_gst_va2-build:latest
```

Reuse it with:

```bash
docker start ovms_gst_va2 >/dev/null 2>&1
docker exec -w /ovms ovms_gst_va2 bash -lc 'echo OK'
```

## Focused Tests

Run process-global VA display tests one at a time.

P7 C-API display-injection proof:

```bash
docker exec -w /ovms ovms_gst_va2 bash -lc \
  "RUN_GPU_TESTS=1 bazel test --config=linux --test_env=RUN_GPU_TESTS=1 \
  --test_output=errors --test_filter='CAPINonCopy.GStreamerDecodesOntoServerOwnedVADisplay' //src:ovms_test"
```

P8 normal server startup:

```bash
docker exec -w /ovms ovms_gst_va2 bash -lc \
  "RUN_GPU_TESTS=1 bazel test --config=linux --test_env=RUN_GPU_TESTS=1 \
  --test_output=errors --test_filter='CAPINonCopy.ServerStartupOwnsConfiguredVaDisplay' //src:ovms_test"
```

P8 graph integration and ModelManager servable resolution, in a separate process:

```bash
docker exec -w /ovms ovms_gst_va2 bash -lc \
  "RUN_GPU_TESTS=1 bazel test --config=linux --test_env=RUN_GPU_TESTS=1 \
  --test_output=errors --test_filter='CAPINonCopy.DeployedGraphUsesModelManagerServable' //src:ovms_test"
```

This test creates a MediaPipe graph in process and passes an `OVMS_Server*`
side packet to the GStreamer C-API inference calculator. It verifies the
shared-display integration, but is not a client-invokable deployed graph. A
production graph must use the OVMS MediaPipe session/inference path rather than
the test-only server-pointer side packet.

## Two Configured Overlay Graphs

`src/test/configs/config_gpu_face_detection_adas_two_overlays.json` deploys two
client-invokable GStreamer graphs in one OVMS process: `gst_overlay_green` and
`gst_overlay_red`. They share the configured `GPU.0` VA display and the
`face_detection_adas` model, while their graph-local side packets select the
overlay color and output file.

Invoke either graph through the KServe `ModelInfer` API with its `OVTENSOR`
`trigger` input. The public response is the matching trigger tensor; the graph
continues its configured finite source through encoder finalization before the
request completes. This is the supported client path for deployed MediaPipe
graphs; the C API inference API invokes models, not MediaPipe graph names.

The focused integration test starts this single configuration, invokes both
graph names, reopens both generated MP4 files, and verifies their respective
green and red overlay chroma values:

```bash
docker exec -w /ovms ovms_gst_va2 bash -lc \
  "RUN_GPU_TESTS=1 bazel test --config=linux --test_env=RUN_GPU_TESTS=1 \
  --test_output=errors --test_filter='CAPINonCopy.ConfiguredOverlayGraphsProduceIndependentOutputs' //src:ovms_test"
```

The full MediaMTX publish/view procedure remains in
[`current_task/P6b_RTSP_Testing_Procedure.md`](../current_task/P6b_RTSP_Testing_Procedure.md).

## Release Image Smoke Test

This validates the packaged release image, not the development build container.
After building a GPU release image, mount models/configuration and pass DRM
access explicitly:

```bash
docker run --rm --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -v "$(pwd)/models":/models:ro \
  -v "$(pwd)/config.json":/config/config.json:ro \
  openvino/model_server:latest-gpu \
  --config_path /config/config.json --port 9000
```

This command is a smoke test to run after producing the release image; it is not
a substitute for the build-container tests above.