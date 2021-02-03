## About this Operator
This Operator is based on a [Helm chart](../../deploy/ovms) for OVMS. It support all the parameters from the helm chart.

It allows for easy deployment and management of OVMS service in the Kubernetes cluster by just creating `Ovms` resource
record.
```bash
kubectl apply -f config/samples/intel_v1alpha1_ovms.yaml
```

## Operator deployment
Deploy the operator using the steps covered in [OperatorHub](https://operatorhub.io)
 
## Using the cluster
OpenVINO Model Server can be consumed as a `Service` with the name matching the `Ovms` record.
```bash
kubectl get pods

kubectl get services
```

## Before you start using the operator
Depending on the deployment configuration there might be pre-requisites for additional records to be created in the cluster. 
Refer to [Helm chart](../../deploy/ovms) parameters documentation.

