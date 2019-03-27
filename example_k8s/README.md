# OpenVINO Model Server example deployment in Kubernetes

## Overview
OpenVINO Model server can be quite easily deployed in Kubernetes which can let scale the inference service horizontally
and ensures high availability.

Below are described simple examples which are using NFS and GCS (Google Cloud Storage) and S3 as the storage for the models.

## Jupyter notebook demo

OpenVINO Model Server deployment in Kubernetes including
the evaluation from gRPC client is demonstrated in the [jupyter notebook demo](OVMS_demo.ipynb). It enables serving of a pre-trained ResNet50 model quantized to INT8 precision.  


## NFS server deployment

There are many possible ways to arrange NFS storage for kubernetes pods. In this example it used a procedure described on 
https://github.com/kubernetes/examples/tree/master/staging/volumes/nfs

It can be applied in AWS, Azure and Google cloud environment. It assumes creating a `persistentvolume` in a form of 
GCE PD or AWS EBS or Azure Disk which is attached to nfs-server pod as a `persistentvolumeclaim`. 

While the nfs-server deployment and service are successfully created a new persistent volume can be created based on NFS
resources which can be applied in all OpenVINO model server pods to serve the IR models. 

```bash
# On GCE (create GCE PD PVC):
$ kubectl create -f examples/staging/volumes/nfs/provisioner/nfs-server-gce-pv.yaml
# On Azure (create Azure Disk PVC):
$ kubectl create -f examples/staging/volumes/nfs/provisioner/nfs-server-azure-pv.yaml
# Common steps after creating either GCE PD or Azure Disk PVC:
$ kubectl create -f examples/staging/volumes/nfs/nfs-server-rc.yaml
$ kubectl create -f examples/staging/volumes/nfs/nfs-server-service.yaml
# get the cluster IP of the server using the following command
$ kubectl describe services nfs-server
# use the NFS server IP to update nfs-pv.yaml and execute the following
$ kubectl create -f examples/staging/volumes/nfs/nfs-pv.yaml
$ kubectl create -f examples/staging/volumes/nfs/nfs-pvc.yaml
```

While those steps are completed there should be present `persistentvolumenclaims` like below:
```bash
kubectl get persistentvolumeclaim
NAME                       STATUS    VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
nfs                        Bound     nfs                                        1Mi        RWX                           23h
nfs-pv-provisioning-demo   Bound     pvc-6de794af-b012-11e8-8916-42010a9c0142   200Gi      RWO            standard       1d
```

With that the `nfs` claim can be applied for other pods in the kubernetes cluster.


## Populating NFS storage with the models
In case the inference models are present on a pod in the cluster they could be populated to the nfs storage by just
mounting it and copying with a structure of folders described on 
[docker_container.md#preparing-the-models](../docs/docker_container.md#preparing-the-models) 

While you have the models available on your laptop just use the command like below:
```bash
kubectl cp /tmp/test_models/saved_models/ [nfs pod name]:/exports/
```

## Deploying OpenVINO Model server with NFS storage

You need to build the OpenVINO Model Server image like described on 
[docker_container.md#building-the-image](../docs/docker_container.md#building-the-image) and push it to a docker 
registry which is accessible in the kuberntes cluster. It could be for example GCR or dockerhub.

Edit a file `openvino_model_server_rc.yaml` and enter the docker image name which was built and pushed earlier.
You should also adjust the command arguments for the OpenVINO Model Server (ie_serving service) according to the
populated models and required configuration. Refer to the documentation on
 [docker_container.md](../docs/docker_container.md) 
 
Now you are ready to deploy the inference service:

```bash
kubectl apply -f openvino_model_server_rc.yaml
kubectl apply -f openvino_model_server_service.yaml
```

A successful attempt should result with a result like below:
```bash
kubectl get pods
NAME                 READY     STATUS    RESTARTS   AGE
nfs-openvino-j9zkr   1/1       Running   0          6s
nfs-openvino-vrngf   1/1       Running   0          6s
nfs-openvino-z2tpk   1/1       Running   0          6h
nfs-server-65t9l     1/1       Running   0          1h

kubectl logs nfs-openvino-j9zkr
2018-09-05 08:11:51,657 - ie_serving.models.model - INFO - Server start loading model: resnet
2018-09-05 08:11:51,661 - ie_serving.models.model - INFO - Creating inference engine object for version: 1
2018-09-05 08:11:52,156 - ie_serving.models.ir_engine - INFO - Matched keys for model: {'outputs': {'resnet_v1_50/predictions/Reshape_1': 'resnet_v1_50/predictions/Reshape_1'}, 'inputs': {'input': 'input'}}
2018-09-05 08:11:52,161 - ie_serving.models.model - INFO - Creating inference engine object for version: 2
2018-09-05 08:11:52,649 - ie_serving.models.ir_engine - INFO - Matched keys for model: {'outputs': {'resnet_v2_50/predictions/Reshape_1': 'resnet_v2_50/predictions/Reshape_1'}, 'inputs': {'input': 'input'}}
2018-09-05 08:11:52,654 - ie_serving.models.model - INFO - List of available versions for resnet model: [1, 2]
2018-09-05 08:11:52,655 - ie_serving.models.model - INFO - Default version for resnet model is 2
2018-09-05 08:11:52,662 - ie_serving.server.start - INFO - Server listens on port 80 and will be serving models: ['resnet']
```

## Deploying OpenVINO Model server with GCS storage

Deployment process with GCS storage is quite similar to the previous one. The same folders structure should be created
in google storage with subfolders representing model versions. Below are example deployment steps which rely on a K8S
secret `gcp-credentials` including key.json file content with GCP authorization key.

```bash
kubectl apply -f openvino_model_server_gs_rc.yaml
kubectl apply -f openvino_model_server_service.yaml
```

Note that in GKE kubernetes cluster the credentials related tags can be dropped as the pods can be authorized natively
using GKE cluster nodes authorization features as long as the models bucket is in the same GCP project with the cluster.

Learn [more about GCP authentication](https://cloud.google.com/docs/authentication/production).


## Deploying OpenVINO Model server with S3 storage

Deployment process with S3 storage requires the same directories structure 
as in previous cases. Example deployment steps are almost identical to those 
from previous cases:

```bash
kubectl apply -f openvino_model_server_s3_rc.yaml
kubectl apply -f openvino_model_server_service.yaml
```

## Readiness checks

By default Kubernetes starts assinging requests to the service pods when they are in `Running` state. In some cases
when the models are stored remotely, more time is needed to download and import the model.

To avoid risk of requests being passed to not fully initialized pods, it is recommended to add Kubernetes readiness checks:

```yaml
    readinessProbe:
      tcpSocket:
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 10
```
The port number should match the one configured as the pod exposed port.

## Testing

When your OpenVINO Model Server is up and running you can start using it. A simple test would be by submitting 
an inference requests using the [exampl client](../example_client)

```bash
python grpc_serving_client.py --grpc_address [external IP address assigned to the service] --grpc_port 80 --model_name resnet --transpose_input True --images_numpy_path 10_imgs.npy --output_name resnet_v1_50/predictions/Reshape_1
Start processing:
	Model name: resnet
	Iterations: 10
	Images numpy path: 10_imgs.npy
	Images in shape: (10, 224, 224, 3)

Top 1: redbone. Processing time: 76.31 ms; speed 13.10 fps
Top 1: redbone. Processing time: 71.17 ms; speed 14.05 fps
Top 1: marmot. Processing time: 74.24 ms; speed 13.47 fps
Top 1: redbone. Processing time: 70.76 ms; speed 14.13 fps
Top 1: marmot. Processing time: 68.66 ms; speed 14.56 fps
Top 1: ice cream, icecream. Processing time: 83.61 ms; speed 11.96 fps
Top 1: redbone. Processing time: 76.49 ms; speed 13.07 fps
Top 1: redbone. Processing time: 67.03 ms; speed 14.92 fps
Top 1: French loaf. Processing time: 67.34 ms; speed 14.85 fps
Top 1: brass, memorial tablet, plaque. Processing time: 68.64 ms; speed 14.57 fps
```
