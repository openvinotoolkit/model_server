# OpenVINO Model Server in Kubernetes {#ovms_docs_kubernetes}

There are three recommended deployment methods of the OpenVINO Model Server in Kubernetes:
- [helm chart](../deploy/README.md) - deploys OVMS instances using [helm](https://helm.sh) package manager for Kubernetes
- Kubernates Operator - manages OVMS using the [Operator](../extras/ovms-operator/README.md)
- For deployments on OpenShift clusters consider using [OpenShift Operator](../extras/openvino-operator-openshift/README.md).  

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_deploy_helm_chart
   ovms_extras_ovms-operator-readme
   ovms_extras_openvino-operator-openshift-readme

@endsphinxdirective