/*
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
*/

package main

import (
	"bytes"
	"encoding/binary"
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	framework "tensorflow/core/framework"
	pb "tensorflow_serving"
	"math"
	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"github.com/nfnt/resize"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
)

// Target model specification
const MODEL_NAME string = "resnet"
const INPUT_NAME string = "data"
const OUTPUT_NAME string = "resnetv17_dense0_fwd"
const IMG_RESIZE_SIZE uint = 224 

// Convert slice of 4 bytes to float32 (assumes Little Endian)
func readFloat32(fourBytes []byte) float32 {
	buf := bytes.NewBuffer(fourBytes)
	var retval float32
	binary.Read(buf, binary.LittleEndian, &retval)
	return retval
}

//change logits array to softmax values
func makeSoftmaxArr(x []float32) []float32 {
    max := x[0]
    for _, v := range x {
        if v > max {
            max = v
        }
    }

    var expSum float32 = 0.0
    exps := make([]float32, len(x))
    for i, v := range x {
        exps[i] = float32(math.Exp(float64(v - max)))
        expSum += exps[i]
    }

    for i := range exps {
        exps[i] /= expSum
    }
    return exps
}

func printPredictionFromResponse(responseProto *framework.TensorProto){
	responseContent := responseProto.GetTensorContent()
	// Get details about output shape
	outputShape := responseProto.GetTensorShape()
	dim := outputShape.GetDim()
	classesNum := dim[1].GetSize()

	//Convert response to array of float32
	outArr := make([]float32, int(classesNum))
	for i := 0; i < int(classesNum); i++ {
		outArr[i] = readFloat32(responseContent[i*4:i*4+4])
	}
	softmaxArr := makeSoftmaxArr(outArr)
	//Get max value and index
	maxVal := softmaxArr[0]
	maxLoc := 0
	for i := 0; i < int(classesNum); i++ {
		if maxVal < softmaxArr[i] {
			maxVal = softmaxArr[i]
			maxLoc = i
		}
	}

	// Get label of the class with the highest confidence
	var label string
	if classesNum == 1000 {
		label = labels[maxLoc]
	} else if classesNum == 1001 {
		label = labels[maxLoc-1]
	} else {
		fmt.Printf("Unexpected class number in the output")
		return 
	}

	fmt.Printf("Predicted class: %s\nClassification confidence: %f%%\n", label, maxVal*100)
}

func run_binary_input(servingAddress string, imgPath string) {
	// Read the image in binary form
	imgBytes, err := ioutil.ReadFile(imgPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Create Predict Request to OVMS
	predictRequest := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          MODEL_NAME,
			SignatureName: "serving_default",
			VersionChoice: &pb.ModelSpec_Version{
				Version: &google_protobuf.Int64Value{
					Value: int64(0),
				},
			},
		},
		Inputs: map[string]*framework.TensorProto{
			INPUT_NAME: &framework.TensorProto{
				Dtype: framework.DataType_DT_STRING,
				TensorShape: &framework.TensorShapeProto{
					Dim: []*framework.TensorShapeProto_Dim{
						&framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
					},
				},
				StringVal: [][]byte{imgBytes},
			},
		},
	}

	// Setup connection with the model server via gRPC
	conn, err := grpc.Dial(servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Cannot connect to the grpc server: %v\n", err)
	}
	defer conn.Close()

	// Create client instance to prediction service
	client := pb.NewPredictionServiceClient(conn)

	// Send predict request and receive response
	predictResponse, err := client.Predict(context.Background(), predictRequest)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println("Request sent successfully")

	// Read prediction results
	responseProto, ok := predictResponse.Outputs[OUTPUT_NAME]
	if !ok {
		log.Fatalf("Expected output: %s does not exist in the response", OUTPUT_NAME)
	}
	
	printPredictionFromResponse(responseProto)
}

func run_with_conversion(servingAddress string, imgPath string) {
	file, err := os.Open(imgPath)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}
	defer file.Close()

	// Decode file to get Image type
	decodedImg, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	// Resize image to match ResNet input
	resizedImg := resize.Resize(IMG_RESIZE_SIZE, IMG_RESIZE_SIZE, decodedImg, resize.Lanczos3)

	// Convert image to gocv.Mat type (HWC layout)
	imgMat, err := gocv.ImageToMatRGB(resizedImg)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	floatMat := gocv.NewMat()
	// Convert type so each value is represented by float32
	// as in Mat generated by ImageToMatRGB values are represented with 8 bit precision
	imgMat.ConvertTo(&floatMat, gocv.MatTypeCV32FC2)

	// Split channels
    channels := gocv.Split(floatMat)
	
	var means = [3]float32{123.675, 116.28, 103.53}
	var scales = [3]float32{58.395, 57.12, 57.375}
    for i := 0; i < 3; i++ {
        // Subtract mean
		meanScalar := gocv.NewScalar(float64(means[i]), 0, 0, 0)
		meanMat := gocv.NewMatWithSizeFromScalar(meanScalar, channels[i].Rows(), channels[i].Cols(), gocv.MatTypeCV32F)
        gocv.Subtract(channels[i], meanMat, &channels[i])
        // Divide by scale
		scaleScalar :=gocv.NewScalar(float64(scales[i]), 0, 0, 0)
		scaleMat := gocv.NewMatWithSizeFromScalar(scaleScalar, channels[i].Rows(), channels[i].Cols(), gocv.MatTypeCV32F)
        gocv.Divide(channels[i], scaleMat, &channels[i])
    }
    // Merge channels back in BGR format
    normalizedMat := gocv.NewMat()
    defer normalizedMat.Close()
    gocv.Merge([]gocv.Mat{channels[2], channels[1], channels[0]}, &normalizedMat)

	// Having right layout and precision, convert Mat to []byte
	imgBytes := normalizedMat.ToBytes()

	// Create Predict Request to OVMS
	predictRequest := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          MODEL_NAME,
			SignatureName: "serving_default",
			VersionChoice: &pb.ModelSpec_Version{
				Version: &google_protobuf.Int64Value{
					Value: int64(0),
				},
			},
		},
		Inputs: map[string]*framework.TensorProto{
			INPUT_NAME: &framework.TensorProto{
				Dtype: framework.DataType_DT_FLOAT,
				TensorShape: &framework.TensorShapeProto{
					Dim: []*framework.TensorShapeProto_Dim{
						&framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(224),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(224),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(3),
						},
					},
				},
				TensorContent: imgBytes,
			},
		},
	}

	conn, err := grpc.Dial(servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Cannot connect to the grpc server: %v\n", err)
	}
	defer conn.Close()

	// Create client instance to prediction service
	client := pb.NewPredictionServiceClient(conn)

	// Send predict request and receive response
	predictResponse, err := client.Predict(context.Background(), predictRequest)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println("Request sent successfully")

	// Read prediction results
	responseProto, ok := predictResponse.Outputs[OUTPUT_NAME]
	if !ok {
		log.Fatalf("Expected output: %s does not exist in the response", OUTPUT_NAME)
	}

	printPredictionFromResponse(responseProto)	
}

func main() {
	servingAddress := flag.String("serving-address", "localhost:8500", "The tensorflow serving address")
	binaryInput := flag.Bool("binary-input", false, "Send JPG/PNG raw bytes")
	flag.Parse()

	if flag.NArg() > 2 {
		fmt.Println("Usage: " + os.Args[0] + " --serving-address localhost:8500 path/to/img")
		os.Exit(1)
	}

	imgPath, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		log.Fatalln(err)
		os.Exit(1)
	}

	if *binaryInput {
		run_binary_input(*servingAddress, imgPath)
	} else {
		run_with_conversion(*servingAddress, imgPath)
	}
}
