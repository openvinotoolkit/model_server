# Deploy Model Server in Kubernetes {#ovms_docs_kubernetes}

There are three recommended methods for deploying OpenVINO Model Server in Kubernetes:
1. [helm chart](../deploy/README.md) - deploys Model Server instances using the [helm](https://helm.sh) package manager for Kubernetes

2. [Kubernetes Operator](../extras/ovms-operator/README.md) - manages Model Server using a Kubernetes Operator
3. [OpenShift Operator](../extras/openvino-operator-openshift/README.md) - manages Model Server instances in Red Hat OpenShift


@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_deploy_helm_chart
   ovms_extras_ovms-operator-readme
   ovms_extras_openvino-operator-openshift-readme

@endsphinxdirective
