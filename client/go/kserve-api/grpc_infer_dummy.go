//
// Copyright (c) 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"time"

	grpc_client "github.com/openvinotoolkit/model_sever/client/go/kserve-api/grpc-client"

	"google.golang.org/grpc"
)

const (
	inputSize  = 10
	outputSize = 10
)

type Flags struct {
	ModelName    string
	ModelVersion string
	URL          string
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "m", "dummy", "Name of model being served. ")
	flag.StringVar(&flags.ModelVersion, "x", "", "Version of model. ")
	flag.StringVar(&flags.URL, "u", "localhost:9000", "Inference Server URL. ")
	flag.Parse()
	return flags
}

func ModelInferRequest(client grpc_client.GRPCInferenceServiceClient, inputData []float32, modelName string, modelVersion string) *grpc_client.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*grpc_client.ModelInferRequest_InferInputTensor{
		&grpc_client.ModelInferRequest_InferInputTensor{
			Name:     "b",
			Datatype: "FP32",
			Shape:    []int64{1, 10},
			Contents: &grpc_client.InferTensorContents{
				Fp32Contents: inputData,
			},
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := grpc_client.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
	}

	// Submit inference request to server
	modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

// Convert slice of 4 bytes to int32 (assumes Little Endian)
func readFloat32(fourBytes []byte) float32 {
	buf := bytes.NewBuffer(fourBytes)
	var retval float32
	binary.Read(buf, binary.LittleEndian, &retval)
	return retval
}


func main() {
	FLAGS := parseFlags()

	// Connect to gRPC server
	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
	}
	defer conn.Close()

	// Create client from gRPC server connection
	client := grpc_client.NewGRPCInferenceServiceClient(conn)

	inputData := make([]float32, inputSize)
	for i := 0; i < inputSize; i++ {
		inputData[i] = float32(i)
	}

	inferResponse := ModelInferRequest(client, inputData, FLAGS.ModelName, FLAGS.ModelVersion)


	outputBytes := inferResponse.RawOutputContents[0]

	outputData := make([]float32, outputSize)
	for i := 0; i < outputSize; i++ {
		outputData[i] = readFloat32(outputBytes[i*4 : i*4+4])
	}

	fmt.Println("\nChecking Inference Outputs\n--------------------------")
	for i := 0; i < outputSize; i++ {
		fmt.Printf("%f => %f\n", inputData[i], outputData[i])
	}
}