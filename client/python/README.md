# Interact with OpenVINO Model Server using Python

OpenVINO Model Server exposes network interface that client applications can interact with.

There's also [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/) package that can be used to send requests to OpenVINO Model Server. 

> **Note**: `tensorflow-serving-api` comes with `tensorflow` dependency which makes the package heavy.

There are sample scripts for `tensorflow-serving-api`:
- [tensorflow-serving-api samples](tensorflow-serving-api/samples)

Additionally, starting with the 2022.2 release, OpenVINO Model Server also supports [KServe API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) calls. You can try it out with the:
 - [KServe samples](kserve-api/samples)

> **Note**: 2025.3 is the last release that includes ovmsclient and it is not supported anymore. If you still need it, use [release 2025.3](https://github.com/openvinotoolkit/model_server/tree/releases/2025/3/client/python/ovmsclient)