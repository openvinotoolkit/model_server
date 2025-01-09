## Deploying Model Server in Kubernetes

There are three recommended methods for deploying OpenVINO Model Server in Kubernetes:
1. [helm chart](https://github.com/openvinotoolkit/operator/tree/main/helm-charts/ovms) - deploys Model Server instances using the [helm](https://helm.sh) package manager for Kubernetes
2. [Kubernetes Operator](https://operatorhub.io/operator/ovms-operator) - manages Model Server using a Kubernetes Operator
3. [OpenShift Operator](https://github.com/openvinotoolkit/operator/blob/main/docs/operator_installation.md#openshift) - manages Model Server instances in Red Hat OpenShift

For operators mentioned in 2. and 3. see the [description of the deployment process](https://github.com/openvinotoolkit/operator/blob/main/docs/modelserver.md)

## Next Steps

- [Start the server](starting_server.md)
- Try the model server [features](features.md)
- Explore the model server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)
