## About this Operator
This Operator is based on a [Helm chart](../../deploy/ovms) for OVMS. It support all the parameters from the helm chart.

It allows for easy deployment and management of OVMS service in the Kubernetes cluster by just creating `Ovms` resource
record.
```bash
kubectl apply -f config/samples/intel_v1alpha1_ovms.yaml
```

## Operator deployment
Deploy the operator using the steps covered in [OperatorHub](https://operatorhub.io)(https://operatorhub.io/operator/ovms-operator)

Alternatively, if you are not using [OLM](https://github.com/operator-framework/operator-lifecycle-manager) component, run commands:
```bash
export IMG=quay.io/openvino/ovms-operator:0.1.0
make install
make deploy IMG=$IMG
```
 
## Using the cluster
OpenVINO Model Server can be consumed as a `Service` with the name matching the `Ovms` record.
```bash
kubectl get pods
NAME                           READY   STATUS    RESTARTS   AGE
ovms-sample-586f6f76df-dpps4   1/1     Running   0          8h

kubectl get services
NAME          TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)             AGE
ovms-sample   ClusterIP   172.25.199.210   <none>        8080/TCP,8081/TCP   8h
```

## Before you start using the operator
Depending on the deployment configuration there might be pre-requisites for additional records to be created in the cluster. 
Refer to [Helm chart](../../deploy/ovms) parameters documentation.
