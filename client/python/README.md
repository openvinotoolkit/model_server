# Interact with OpenVINO Model Server using Python

OpenVINO Model Server exposes network interface that client applications can interact with. Such interaction is really simple in Python with [ovmsclient](https://pypi.org/project/ovmsclient/) package available on PyPi. 
[Learn more about ovmsclient](ovmsclient/lib)

There's also [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/) package that can be used to send requests to OpenVINO Model Server. 

> **Note**: `tensorflow-serving-api` comes with `tensorflow` dependency which makes the package heavy. For a lightweight solution consider using `ovmsclient`.

There are sample scripts for both solutions:
- [ovmsclient samples](ovmsclient/samples)
- [tensorflow-serving-api samples](tensorflow-serving-api/samples)

Additionally, starting with the 2022.2 release, OpenVINO Model Server also supports [KServe API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) calls. You can try it out with the:
 - [KServe samples](kserve-api/samples)

