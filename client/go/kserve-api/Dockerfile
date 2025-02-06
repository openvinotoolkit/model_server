#
# Copyright (c) 2022 Intel Corporation
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
FROM golang:1.21.9

RUN apt-get update && \
    apt-get -y install git unzip build-essential autoconf libtool protobuf-compiler libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34
RUN go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.4.0

# Compile API
RUN wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2025/0/src/kfserving_api/grpc_predict_v2.proto
RUN echo 'option go_package = "./grpc-client";' >> grpc_predict_v2.proto
RUN protoc --go_out="./" --go-grpc_out="./" ./grpc_predict_v2.proto

COPY . /app
WORKDIR /app/build
RUN go get google.golang.org/grpc@v1.62.0 
RUN go get google.golang.org/grpc/internal/transport@v1.62.0
RUN for I in ../*go; do go build $I; done
# In case of dependencies version updates below might be useful, so we can substitute the go.mod and go.sum files
# with versions that has been processed by go. 
# RUN cp /app/go.mod /app/go.sum /app/build 