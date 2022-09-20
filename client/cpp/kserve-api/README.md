# KServe API usage samples

OpenVINO Model Server 2022.2 release introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

## Before you run the samples

### Clone OpenVINO&trade; Model Server GitHub repository and enter model_server directory.
```Bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/cpp/kserve-api
```

### Build client library and samples
```Bash
cmake . && make
```