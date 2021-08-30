FROM golang:latest

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