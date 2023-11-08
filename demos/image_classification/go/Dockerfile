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

FROM golang:1.20.10
ARG http_proxy

RUN echo "Acquire::http::Proxy \"$http_proxy\";" > /etc/apt/apt.conf.d/proxy.conf
RUN apt-get update && \
    apt-get -y install git unzip build-essential autoconf libtool protobuf-compiler libdc1394-dev libprotobuf-dev

RUN apt-get update && apt-get install -y sudo

RUN rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/hybridgroup/gocv.git && \
    cd gocv && \
    sed -i "s:libdc1394-22-dev:libdc1394-dev:g; s:libtbb2:libtbb12:g" Makefile && \
    make install

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.31
RUN go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.3

# Compile API
RUN protoc -I apis/ apis/tensorflow_serving/apis/*.proto --go_out="./" --go-grpc_out="./"
RUN protoc -I apis/ apis/tensorflow/core/framework/*.proto --go_out="./" --go-grpc_out="./"

# Move compiled protos under GOROOT
RUN mv tensorflow /usr/local/go/src/
RUN mv tensorflow_serving /usr/local/go/src/

RUN go mod tidy
RUN go build .

ENTRYPOINT ["/app/ovmsclient"]
