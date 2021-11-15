#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM golang:latest
ARG http_proxy

RUN echo "Acquire::http::Proxy \"$http_proxy\";" > /etc/apt/apt.conf.d/proxy.conf
RUN apt-get update && \
    apt-get -y install git unzip build-essential autoconf libtool protobuf-compiler libprotobuf-dev
RUN go get google.golang.org/grpc
RUN go get github.com/golang/protobuf/protoc-gen-go

# Install Go OpenCV (to simplify postprocessing)
RUN apt-get install -y sudo && \
    git clone https://github.com/hybridgroup/gocv.git && \
    cd gocv && \
    make install

RUN mkdir /app
COPY . /app
WORKDIR /app

# Compile API
RUN protoc -I apis/ apis/tensorflow_serving/apis/*.proto --go_out=plugins=grpc:.
RUN protoc -I apis/ apis/tensorflow/core/framework/*.proto --go_out=plugins=grpc:.

# Move compiled protos under GOROOT
RUN mv tensorflow /usr/local/go/src/
RUN mv tensorflow_serving /usr/local/go/src/

## we run go build to compile the binary
RUN go mod init ovmsclient
RUN go mod tidy
RUN go build .


ENTRYPOINT ["/app/ovmsclient"]