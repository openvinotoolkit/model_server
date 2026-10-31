# Standalone MediaPipe + MPI

This example runs OV-only GStreamer inference without the OVMS server. The
MediaPipe graph reads a video through MPI, runs OpenVINO inference, and prints
the detection count for each frame.

## Selective Bazel Dependencies

`//standalone:mpi_openvino_video` depends on MPI, the OV-only video inference
calculator, OpenVINO, and MediaPipe. It does not depend on `//src:ovms_lib` or
`//src:ovms`. The target dependency list in `BUILD` is the project setup
example: consumer targets should depend on these components instead of the
OVMS server target.

This package is built inside the OVMS repository workspace. It demonstrates
selective Bazel targets, but is not yet a separately importable external
workspace; the repository's monolithic `WORKSPACE` still needs to be split for
that workflow.

## Build and Run

Build in the configured Linux build container, which has GStreamer and its
plugins:

```bash
bazel build --config=mp_on //standalone:mpi_openvino_video
```

Run with a video and an OpenVINO IR XML file. Width and height default to
672x384 and can be overridden together:

```bash
bazel run --config=mp_on //standalone:mpi_openvino_video -- \
  standalone/openvino_video_inference.pbtxt \
  recording_3_raw.avi \
  /path/to/face-detection-adas-0001.xml
```

The host needs the matching `.bin` weights file. The runtime also needs a
working GStreamer installation with the plugins required to decode the input
video, plus GPU/VA access for this calculator's Linux path.

## Verify Layering

Run the dependency check from the repository root:

```bash
bash standalone/check_dependencies.sh
```

It uses configured `bazel cquery` paths to ensure the executable cannot reach
either OVMS server target. Override the Bazel config with `BAZEL_CONFIG` if
needed.