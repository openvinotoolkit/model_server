FROM quay.io/operator-framework/helm-operator:v1.3.0
LABEL "Description"="This OpenVINO(TM) Model Server Operator"
LABEL "Vendor"="Intel Corporation"
ENV HOME=/opt/helm
COPY watches.yaml ${HOME}/watches.yaml
COPY helm-charts  ${HOME}/helm-charts
COPY LICENSE LICENSE
WORKDIR ${HOME}
