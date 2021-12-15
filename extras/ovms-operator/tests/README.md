# Operator testing procedure

## Building the operator image 

```
export REGISTRY=quay.io
export IMG=$REGISTRY/ovms-operator:latest
make docker-build docker-push IMG=$IMG
```

## Deploy in Kubernetes the operator without OLM
```bash
export IMG=quay.io/openvino/ovms-operator:0.1.0
make install
make deploy IMG=$IMG
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
git clone https://github.com/operator-framework/community-operators
cd community-operators/upstream-community-operators/ovms-operator/0.1.0

# edit manifests/ovms-operator.clusterserviceversion.yaml and set image: to point to the test operator image

docker build -t $IMG_BUNDLE .
docker push $IMG_BUNDLE
sudo opm index add --bundles $IMG_BUNDLE --from-index quay.io/operatorhubio/catalog:latest -c docker --tag $IMG_CATALOG
or
opm index add --bundles $IMG_BUNDLE --from-index registry.redhat.io/redhat/community-operator-index:v4.7 -p podman --tag $IMG_CATALOG
docker push $IMG_CATALOG
```

## Setting up test k8s cluster
```bash 
apt install -y kubeadm kubelet
kubeadm reset -f
kubeadm init --pod-network-cidr=10.244.0.0/16
cp -f /etc/kubernetes/admin.conf $HOME/.kube/config
kubectl apply -f https://github.com/coreos/flannel/raw/master/Documentation/kube-flannel.yml
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl get pod --all-namespaces
echo "cluster installed"
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