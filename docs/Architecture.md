# OpenVINO&trade; Model Server Architecture

- OpenVINO&trade; Model Server is a C++ implementation of gRPC and RESTful API interfaces defined by [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving).

- OpenVINO&trade; Model Server uses [Inference Engine](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_inference_engine_intro.html) libraries from OpenVINO&trade; toolkit in the backend, which speeds up the execution on CPU and enables it on AI accelerators like [Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html), iGPU(Integrated Graphics Processing Unit), [HDDL](https://docs.openvinotoolkit.org/2018_R5/_docs_IE_DG_supported_plugins_HDDL.html) and FPGAs.

- API requests in gRPC code skeleton are created based on [TensorFlow Serving Core Framework](https://www.tensorflow.org/tfx/guide/serving) with tunned implementation of requests handling.

- Services are designed via set of C++ classes managing AI models in Intermediate Representation format. [OpenVINO&trade; Inference Engine](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_inference_engine_intro.html) component executes the graphs operations.

![architecture chart](./Images/serving-c.png)

**Figure 1: Docker Container (VM or Bare Metal Host)**

- OpenVINO&trade; Model Server requires the models to be present in the local file system or they could be hosted remotely on object storage services. Both Google Cloud Storage and S3 compatible storage are supported. Refer to [Preparing the Models Repository](./PreparingModelsRepository.md) for more details.

- OpenVINO&trade; Model Server is suitable for landing in Kubernetes environment. It can be also hosted on a bare metal server, virtual machine or inside a docker container. It is also suitable for landing in Kubernetes environment. 

- The only two exposed network interfaces are [gRPC](./ModelServerGRPCAPI.md) and [RESTful API](./ModelServerRESTAPI.md), which currently _does not_ include authorization, authentication, or data encryption. Those functions are expected to be implemented outside of the model server.