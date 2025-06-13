# Single Face Analysis Pipeline Demo {#ovms_demo_single_face_analysis_pipeline}

This document presents a models ensemble as an example of [DAG Scheduler](../../../docs/dag_scheduler.md) implementation.
It describes how to combine several models to perform multiple inference operations with a single prediction call.
When you need to execute several predictions on the same data, you can create a pipeline, which combines the results from several models.

![diagram](single_face_analysis_pipeline.png)

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip 

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

## Prepare models

In this example the following models are used:

[age-gender-recognition-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/intel/age-gender-recognition-retail-0013/README.md)

[emotions-recognition-retail-0003](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/intel/emotions-recognition-retail-0003/README.md)

Clone the repository and enter single_face_analysis_pipeline directory
```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/single_face_analysis_pipeline/python
```

You can prepare the workspace that contains all the above by just running


You can prepare the workspace that contains all the above by running

```console
	curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml -o workspace/age-gender-recognition-retail-0013/1/age-gender-recognition-retail-0013.xml
	curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin -o workspace/age-gender-recognition-retail-0013/1/age-gender-recognition-retail-0013.bin
	curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -o workspace/emotions-recognition-retail-0003/1/emotions-recognition-retail-0003.xml
	curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -o workspace/emotions-recognition-retail-0003/1/emotions-recognition-retail-0003.bin
```

### Final directory structure

You should have `workspace` directory ready with the following content.
```console
workspace/
├── age-gender-recognition-retail-0013
│   └── 1
│       ├── age-gender-recognition-retail-0013.bin
│       └── age-gender-recognition-retail-0013.xml
└── emotions-recognition-retail-0003
    └── 1
        ├── emotions-recognition-retail-0003.bin
        └── emotions-recognition-retail-0003.xml
```

## Server Deployment
:::{dropdown} **Deploying with Docker**
```bash
cp config.json workspace/.
chmod -R 755 workspace
docker run -p 9000:9000 -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json --port 9000
```
:::
:::{dropdown} **Deploying on Bare Metal**
Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.
```bat
copy config.json workspace
ovms --config_path workspace/config.json --port 9001
```
:::
## Requesting the Service

Exemplary client [single_face_analysis_pipeline.py](https://github.com/openvinotoolkit/model_server/blob/releases/2025/2/demos/single_face_analysis_pipeline/python/single_face_analysis_pipeline.py) can be used to request pipeline deployed in previous step.

```console
pip3 install -r requirements.txt
``` 

Now you can create directory for text images and run the client:

```console
python single_face_analysis_pipeline.py --image_path ../../common/static/images/faces/face1.jpg --grpc_port 9000
Age results: [[[21.099792]]]
Gender results: Female: 0.9483401 ; Male: 0.051659837
Emotion results: Natural: 0.02335789 ; Happy: 0.9449672 ; Sad: 0.001236845 ; Surprise: 0.028111042 ; Angry: 0.0023269346
```

### Next step

For more advanced use case with extracting and analysing multiple faces on the same image see [multi_faces_analysis_pipeline](../../multi_faces_analysis_pipeline/python/README.md) demo.