package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	framework "tensorflow/core/framework"
	pb "tensorflow_serving"

	"io/ioutil"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
)

func main() {
	servingAddress := flag.String("serving-address", "localhost:8500", "The tensorflow serving address")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Println("Usage: " + os.Args[0] + " --serving-address localhost:8500 path/to/img")
		os.Exit(1)
	}

	imgPath, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		log.Fatalln(err)
		os.Exit(1)
	}

	// Read the image in binary form
	imageBytes, err := ioutil.ReadFile(imgPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Target model specification
	const MODEL_NAME string = "resnet"
	const INPUT_NAME string = "map/TensorArrayStack/TensorArrayGatherV3"
	const OUTPUT_NAME string = "softmax_tensor"

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
				StringVal: [][]byte{imageBytes},
			},
		},
	}

	// Setup connection with the model server via gRPC
	conn, err := grpc.Dial(*servingAddress, grpc.WithInsecure())
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
	responseProto := predictResponse.Outputs[OUTPUT_NAME]
	responseContent := responseProto.GetTensorContent()

	// Get details about output shape
	outputShape := responseProto.GetTensorShape()
	dim := outputShape.GetDim()
	classesNum := dim[1].GetSize()

	// Convert bytes to matrix
	outMat, err := gocv.NewMatFromBytes(1, int(classesNum), gocv.MatTypeCV32FC1, responseContent)
	outMat = outMat.Reshape(1, 1)

	// Find maximum value along with its index in the output
	_, maxVal, _, maxLoc := gocv.MinMaxLoc(outMat)

	// Get label of the class with the highest confidence
	var label string
	if classesNum == 1000 {
		label = labels[maxLoc.X]
	} else if classesNum == 1001 {
		label = labels[maxLoc.X-1]
	} else {
		fmt.Printf("Unexpected class number in the output")
		return
	}

	fmt.Printf("Predicted class: %s\nClassification confidence: %f%%\n", label, maxVal*100)
}
