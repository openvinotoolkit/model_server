---
id: ovms_docs_advanced
sidebar_position: 3
---

# Advanced Features

:::info Phase 1 structure note (pending stakeholder review)
Landing page replacing the old 'Advanced' catch-all; links to src/example/SampleCpuExtension and extras/nginx-mtls-auth left in place (out of Phase 1 move scope, same treatment as demos/).
:::


## CPU Extensions
Implement any CPU layer, that is not supported by OpenVINO yet, as a shared library.

[Learn more](../../src/example/SampleCpuExtension/README.md)

## Model Cache
Leverage the OpenVINO [model caching](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) feature to speed up subsequent model loading on a target device.

[Learn more](../deploy-and-operate/model-cache.md)

## Custom Model Loader
Write your own custom model loader based on a predefined interface and load it similar to a dynamic library.

[Learn more](custom-model-loader.md)

## Securing Model Server with NGINX
Protect network endpoints with traffic encryption and client authorization using a reverse proxy

[Learn more](../../extras/nginx-mtls-auth/README.md)