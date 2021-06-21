# Operator testing procedure

## Building the operator image 

```bash
export REGISTRY=quay.io
export IMG=$REGISTRY/ovms-operator:latest
make docker-build docker-push IMG=$IMG
```

## Building the test olm catalog

Prerequisites:
- podman
- [opm](https://github.com/operator-framework/operator-registry/releases/latest) 
- registry with push permissions

```bash
export REGISTRY=quay.io
export IMG_BUNDLE=$REGISTRY/ovms-operator-bundle:latest
export IMG_CATALOG=$REGISTRY/test-catalog:latest
cd model_server/extras/openvino-operator-openshift/openshift-bundle/0.2.0

# edit manifests/ovms-operator.clusterserviceversion.yaml and set image: to point to the test operator image

podman build -t $IMG_BUNDLE .
podman push $IMG_BUNDLE
opm index add --bundles $IMG_BUNDLE --from-index registry.redhat.io/redhat/community-operator-index:v4.7 -p podman --tag $IMG_CATALOG
podman push $IMG_CATALOG
```

## Deployment of the OVMS operator
Prerequisites:
- [operator-sdk](https://github.com/operator-framework/operator-sdk)

```bash
echo "installing olm"
operator-sdk olm install
operator-sdk olm status

echo "add test catalog"
kubectl apply -f catalog-source.yaml

kubectl get pod -n olm
kubectl get packagemanifests | grep ovms-operator
echo "test catalog installed"

kubectl create ns operator
kubectl apply -f operator-group.yaml
kubectl apply -f operator-subscription.yaml

kubectl get clusterserviceversion --all-namespaces
```

## Testing the operator usage
```bash
kubectl create ns ovms
kubectl apply -f sample1.yaml
kubectl apply -f sample2.yaml
```