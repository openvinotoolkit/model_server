# Streaming API (preview) {#ovms_docs_streaming_endpoints}

## Introduction
OpenVINO Model Server implements gRPC KServe extension which adds separate RPC for [bidirectional streaming](https://grpc.io/docs/what-is-grpc/core-concepts/#bidirectional-streaming-rpc) use cases. It means that besides unary RPC, where client sends a single request and gets back a single response, client is able to initiate connection and send/receive messages in any number and order using `ModelStreamInfer` procedure:

```
service GRPCInferenceService
{
  ...
  // Standard
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse) {}
  // Extension
  rpc ModelStreamInfer(stream ModelInferRequest) returns (stream ModelStreamInferResponse) {}
  ...
}
```

This becomes very useful for serving [MediaPipe Graphs](./mediapipe.md). In unary inference RPC, each request is using separate and independent MediaPipe graph instance.
However, in streaming inference RPC MediaPipe graph is created only once per connection, and is reused by subsequent requests to the same gRPC stream. This avoids graph initialization overhead and increases its overall throughput.

![diagram](streaming_diagram.svg)

## Graph Selection
After opening a stream, the first gRPC request defines which graph definition will be selected for execution (`model_name` proto fields).
Afterwards, subsequent requests are required to match the servable name and version, otherwise the error is reported and input packets are not pushed to the graph. However, the graph remains available for correct requests.

> NOTE: The server closes the stream after the first request if requested graph is non-existent or retired.

## Timestamping
MediaPipe Graphs require packets to include timestamp information for synchronization purposes. Each input stream in the graph requires timestamps to be monotonically increasing. Read further about [MediaPipe timestamping](https://developers.google.com/mediapipe/framework/framework_concepts/synchronization#timestamp_synchronization).

### Automatic timestamping
By default OpenVINO Model Server assigns timestamps automatically. Each gRPC request is treated as separate point on the timeline, starting from 0. Each request is deserialized sequentially and increases the timestamp by 1.

> NOTE: It means that in order to send multiple inputs with the same timestamp, client needs to pack it into single request (or use manual timestamping).

### Manual timestamping
Optionally, the client is allowed to include timestamp manually via request parameter `OVMS_MP_TIMESTAMP`. It is applied to all the packets deserialized from the request.

It is possible to mix manual/automatic timestamping. After correct deserialization step, default automatic timestamp is always equal to `previous_timestamp + 1`.

## Preserving State Between Requests
Note that subsequent requests in a stream have access to the same instance of MediaPipe graph. It means that it is possible implement graph that saves intermediate state and act in stateful manner. It might be an advantage f.e. for object tracking use cases.

## Useful links
- example client snippets TODO
- [complete demo with streaming](../demos/mediapipe/holistic_tracking/README.md)

> NOTE: Streaming API does not support requesting single models nor DAGs (Directed Acyclic Graphs) - those need to be included in MediaPipe Graph in order to use streaming.

