# Interact with OpenVINO Model Server using Python

OpenVINO Model Server exposes network interface that client applications can interact with. Such interaction is really simple in Python with [ovmsclient](https://pypi.org/project/ovmsclient/) package available on PyPi. 
[Learn more about ovmsclient](ovmsclient/lib)

There's also [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/) package that can be used to send requests to OpenVINO Model Server. 

> **Note**: `tensorflow-serving-api` comes with `tensorflow` dependency which makes the package heavy. For a lightweight solution consider using `ovmsclient`.

There are sample scripts for both solutions:
- [ovmsclient samples](ovmsclient/samples)
- [tensorflow-serving-api samples](tensorflow-serving-api/samples)

