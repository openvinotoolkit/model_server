# Cloud Native Quickstart Guide {#ovms_docs_cloud_native_quick_start_guide}

By following this guide, user can deploy OpenVINO Model Server as a Kubernetes Pod and store trained AI model in `Persistent Volume`(PV) and `Persistent Volume_Claim`(PVC) to perform inference using pre-trained models in either [OpenVINO IR](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html), [ONNX](https://onnx.ai/), [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) or [TensorFlow](https://www.tensorflow.org/) format. This guide uses a [Faster R-CNN with Resnet-50 V1 Object Detection model](https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1) in TensorFlow format.

> **Note**: - OpenVINO Model Server can run on Linux and macOS. For use on Windows, [WSL](https://docs.microsoft.com/en-us/windows/wsl/) is required.

To quickly start using Cloud Native OpenVINO™ Model Server follow these steps:
1. Install Kubernetes Cluster
2. Install OpenVINO Toolkit Operator
3. Provide a model in PV/PVC
4. Deploy an OpenVINO Model Server Instance
5. Verify OpenVINO Model Serving by sending Inference Request

### Step 1: Install Kubernetes Cluster

There are many methods to install a Kubernetes Cluster, such as `kind`, `minikube` and `kubeadm`. This guide is using [Creating a cluster with kubeadm](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/) to create a Kubernetes cluster.

### Step 2: Install OpenVINO Toolkit Operator

This Kubernetes operator named [OpenVINO Toolkit Operator](https://operatorhub.io/operator/ovms-operator) manages OpenVINO components in Kubernetes, and currently available components are ModelServer. Please follow instructions:

- 1. Install Operator Lifecycle Manager (OLM) on your cluser:

```bash
$ curl -sL https://github.com/operator-framework/operator-lifecycle-manager/releases/download/v0.27.0/install.sh | bash -s v0.27.0
```

- 2. Verify the OLM on your cluster:

```bash
$ kubectl get pods -n olm
NAME                                READY   STATUS    RESTARTS   AGE
catalog-operator-74944d49c6-68257   1/1     Running   0          5m
olm-operator-5c789d6974-4w2sz       1/1     Running   0          5m
operatorhubio-catalog-jwmdp         1/1     Running   0          5m
packageserver-6d5b65f789-24dql      1/1     Running   0          5m
packageserver-6d5b65f789-7bbk5      1/1     Running   0          5m
```

- 3. Install the operator by running the following command:

```bash
$ kubectl create -f https://operatorhub.io/install/ovms-operator.yaml
```

- 4. Verify the operator by using below command:

```bash
$ kubectl get csv -n operators
NAME                       DISPLAY                     VERSION   REPLACES                   PHASE
openvino-operator.v1.1.0   OpenVINO Toolkit Operator   1.1.0     openvino-operator.v1.0.0   Succeeded

$ kubectl get pods -n operators
NAME                                                    READY   STATUS    RESTARTS   AGE
openvino-operator-controller-manager-548888ddc7-jq4mh   2/2     Running   0          5m
```

### Step 3: Provide a model in PV/PVC

As an end-to-end example, this guide uses `PV/PVC` to store the AI model, and user can also store their model by using [Cloud Storage](using_cloud_storage.md). Before the storage of AI model, please refer to [Model Repository](models_repository.md) to know how to organize components of the model. 

- 1. Pulling an object detection model from TensorFlow Hub:

```bash
$ mkdir -p /models/model/1
$ wget https://storage.googleapis.com/tfhub-modules/tensorflow/faster_rcnn/resnet50_v1_640x640/1.tar.gz
$ tar xzf 1.tar.gz -C /models/model/1
```

OpenVINO Model Server expects a particular folder structure for models - in this case `model` directory has the following content: 

```bash
models
└── model
    └── 1
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index
```

Sub-folder `1` indicates the version of the model. If you want to upgrade the model, other versions can be added in separate subfolders (2,3...). 
For more information about the directory structure and how to deploy multiple models at a time, check out the following sections:
- [Preparing models](models_repository.md)
- [Serving models](starting_server.md)
- [Serving multiple model versions](model_version_policy.md) 

- 2. Create the PV/PVC to mount the model directory

The PV/PVC content of yaml file `pv-and-pvc.yaml` is below:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: "my-models-pv"
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  hostPath:
    path: "/models"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: "my-models-pvc"
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
```

Then apply this yaml file by below command:

```bash
$ kubectl apply -f pv-and-pvc.yaml
```

> **Note**: - Please make sure the namespace of PV/PVC is same with the OpenVINO Model Server Instance deployed in following step.

- 3. Set the model directory permission

We must set the model directory permission of `my-models-pvc` due to the container of OpenVINO Model Server uses the security context of account `ovms` with pid 5000 and group 5000 by below command:

```bash
$ sudo chown -R 5000:5000 /models
$ sudo chmod -R 755 /models
```

### Step 4: Deploy an OpenVINO Model Server Instance

There should be CRD of OpenVINO Model Server after `Step 2` completed. Therefore, user can deploy an OpenVINO Model Server Instance based on below yaml file named `openvino_model_server_sample.yaml`

```yaml
apiVersion: intel.com/v1alpha1
kind: ModelServer
metadata:
  name: model-server-sample
spec:
  image_name: 'openvino/model_server:latest'
  deployment_parameters:
    replicas: 1
  service_parameters:
    grpc_port: 9000
    rest_port: 8000
  models_settings:
    single_model_mode: true
    config_configmap_name: ''
    config_path: ''
    model_name: faster_rcnn
    model_path: '/models/model'
    nireq: 0
    plugin_config: '{"PERFORMANCE_HINT":"LATENCY"}'
    batch_size: ''
    shape: ''
    model_version_policy: '{"latest": { "num_versions":1 }}'
    layout: ''
    target_device: CPU
    is_stateful: false
    idle_sequence_cleanup: false
    low_latency_transformation: true
    max_sequence_number: 0
  server_settings:
    file_system_poll_wait_seconds: 0
    sequence_cleaner_poll_wait_minutes: 0
    log_level: DEBUG
    grpc_workers: 1
    rest_workers: 2 
  models_repository:
    https_proxy: ''
    http_proxy: ''
    storage_type: cluster
    models_host_path: ''
    models_volume_claim: 'my-models-pvc'
    aws_secret_access_key: ''
    aws_access_key_id: ''
    aws_region: ''
    s3_compat_api_endpoint: ''
    gcp_creds_secret_name: ''
    azure_storage_connection_string: ''
```

Then apply this yaml file by below command:

```bash
$ kubectl apply -f openvino_model_server_sample.yaml
```

Verify the OpenVINO Model Server Instance by below command:

```bash
$ kubectl get pods
NAME                                        READY   STATUS    RESTARTS   AGE
model-server-sample-ovms-78f6b96ffb-plrp8   1/1     Running   0          11m

$ kubectl get svc
NAME                       TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)             AGE
kubernetes                 ClusterIP   10.80.0.1      <none>        443/TCP             27h
model-server-sample-ovms   ClusterIP   10.82.176.69   <none>        9000/TCP,8000/TCP   11m
```

> **Note**: - This guide use PV/PVC to store the model, therefore, `models_repository.models_volume_claim` is set to Persistent Volume Claim `my-models-pvc` created above, and `models_settings` is set to `/models/model` which is mounted by Persistent Volume. Please refer to [Model Server Parameters](parameters.md) for more detail.

### Step 5: Verify OpenVINO Model Serving by Running Inference

User can deploy a client test pod inside the same Kubernetes cluster to run inference to verify OpenVINO Model Serving by following instructions:

- 1. Prepare a client test pod for verification

Launch the `client-test` pod:

```bash
$ kubectl create deployment client-test --image=python:3.8.13 -- sleep infinity
```

- 2. Verify the OpenVINO Model Server from the client test pod

```bash
$ kubectl exec -it $(kubectl get pod -o jsonpath="{.items[0].metadata.name}" -l app=client-test) -- bash
$ curl http://model-server-sample-ovms:8000/v1/config
{
  "faster_rcnn": {
    "model_version_status": [
      {
        "version": "1",
        "state": "AVAILABLE",
        "status": {
          "error_code": "OK",
          "error_message": "OK"
        }
      }
    ]
  }
}
```

Above output indicate that the model server is working well.

- 3. Running inference inside the client test pod 

Prepare the Example Client Components in local file system:

```bash
$ mkdir -p ~/ecc && cd ~/ecc
$ wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/object_detection/python/object_detection.py
$ wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/object_detection/python/requirements.txt
$ wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_91cl.txt
$ wget https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg
```

Copy the Example Client Components into client-test pod:

```bash
$ kubectl cp -n default . $(kubectl get pod -o jsonpath="{.items[0].metadata.name}" -l app=client-test):/root
$ kubectl exec -it $(kubectl get pod -o jsonpath="{.items[0].metadata.name}" -l app=client-test) -- ls -ls /root
  4 -rw-r--r-- 1 1000 1000    702 Apr 10 09:03 coco_91cl.txt
184 -rw-r--r-- 1 1000 1000 186584 Apr 10 09:03 coco_bike.jpg
  8 -rw-r--r-- 1 1000 1000   6182 Apr 10 09:03 object_detection.py
268 -rw-r--r-- 1 root root 271313 Apr 11 07:10 output.jpg
  4 -rw-r--r-- 1 1000 1000     44 Apr 10 09:03 requirements.txt
```

Go to the folder with the Example Client Components and install dependencies:

```bash
$ kubectl exec -it $(kubectl get pod -o jsonpath="{.items[0].metadata.name}" -l app=client-test) -- bash
$ cd /root
$ pip install --upgrade pip
$ pip install -r requirements.txt

$ python3 object_detection.py --image coco_bike.jpg --output output.jpg --service_url model-server-sample-ovms:9000
$ ls -ls
total 464
  4 -rw-r--r-- 1 1000 1000    702 Apr 10 09:03 coco_91cl.txt
184 -rw-r--r-- 1 1000 1000 186584 Apr 10 09:03 coco_bike.jpg
  8 -rw-r--r-- 1 1000 1000   6182 Apr 10 09:03 object_detection.py
264 -rw-r--r-- 1 root root 270070 Apr 11 10:08 output.jpg
  4 -rw-r--r-- 1 1000 1000     44 Apr 10 09:03 requirements.txt
```

> **Note**: - Please make sure the `HTTP_PROXY` and `HTTPS_PROXY` are set correctly to install the dependencies libs in `requirements.txt`.

- 4. Review the Results

Showing as previous instruction, User can find output file named `output.jpg` containing inference results. In this case, it should be a modified input image with bounding boxes indicating detected objects and their labels.

![Inference results](cloud_native_quickstart_result.jpeg)

> **Note**: Similar steps can be performed with other model formats. Check the [ONNX use case example](../demos/using_onnx_model/python/README.md), 
[TensorFlow classification model demo](../demos/image_classification_using_tf_model/python/README.md ) or [PaddlePaddle model demo](../demos/segmentation_using_paddlepaddle_model/python/README.md).

Congratulations, you have completed the Cloud Native QuickStart guide. Try other Model Server [demos](../demos/README.md) or explore more [features](features.md) to create your application.
