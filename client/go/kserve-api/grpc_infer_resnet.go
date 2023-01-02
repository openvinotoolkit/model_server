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
	"os"
	"bufio"
	"strings"
	"strconv"
	"io/ioutil"

	grpc_client "github.com/openvinotoolkit/model_sever/client/go/kserve-api/grpc-client"

	"google.golang.org/grpc"
)

type Flags struct {
	ModelName    string
	ModelVersion string
	URL          string
	Images		 string
	Labels		 string
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "n", "resnet", "Name of model being served. ")
	flag.StringVar(&flags.ModelVersion, "v", "", "Version of model. ")
	flag.StringVar(&flags.URL, "u", "localhost:9000", "Inference Server URL. ")
	flag.StringVar(&flags.Images, "i", "", "Path to a file with a list of labeled images.")
	flag.StringVar(&flags.Labels, "l", "", "Path to a file with a list of labels.")
	flag.Parse()
	return flags
}

func ModelInferRequest(client grpc_client.GRPCInferenceServiceClient, fileName string, modelName string, modelVersion string) *grpc_client.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*grpc_client.ModelInferRequest_InferInputTensor{
		&grpc_client.ModelInferRequest_InferInputTensor{
			Name:     "0",
			Datatype: "BYTES",
			Shape:    []int64{1},
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := grpc_client.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
	}

    bytes, err := ioutil.ReadFile(fileName)


	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, bytes) 

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

	readFile, err := os.Open(FLAGS.Labels)

    if err != nil {
        fmt.Println(err)
    }

    fileScanner := bufio.NewScanner(readFile)
 
    fileScanner.Split(bufio.ScanLines)

	var labels []string
    for fileScanner.Scan() {
		labels = append(labels, fileScanner.Text())
	}
	readFile.Close()

	readFile, err = os.Open(FLAGS.Images)
  
    if err != nil {
        fmt.Println(err)
    }
    fileScanner = bufio.NewScanner(readFile)
 
    fileScanner.Split(bufio.ScanLines)
  
    for fileScanner.Scan() {
		line := strings.Split(fileScanner.Text(), " ")
		inferResponse := ModelInferRequest(client, line[0], FLAGS.ModelName, FLAGS.ModelVersion)

		outputBytes := inferResponse.RawOutputContents[0]

		outputData := make([]float32, 1000)
		for i := 0; i < 1000; i++ {
			outputData[i] = readFloat32(outputBytes[i*4 : i*4+4])
		}

		max := outputData[0]
		maxi := -1
		for i :=1; i < 1000; i++ {
			if max < outputData[i] {
				max = outputData[i]
				maxi = i
			}
		}
		lb, err := strconv.Atoi(line[1])
		if err != nil{
			log.Fatalf("Error processing InferRequest: %v", err)
		}
		if maxi == lb {
			fmt.Printf("%s classified as %d %s\n", line[0], maxi, labels[maxi])
		} else{
			fmt.Printf("%s classified as %d %s, should be %s %s\n", line[0], maxi, labels[maxi], line[1], labels[lb])
		}

    }
  
    readFile.Close()
}