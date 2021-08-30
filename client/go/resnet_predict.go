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

	imageBytes, err := ioutil.ReadFile(imgPath)
	if err != nil {
		log.Fatalln(err)
	}

	predictRequest := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          "resnet",
			SignatureName: "serving_default",
			VersionChoice: &pb.ModelSpec_Version{
				Version: &google_protobuf.Int64Value{
					Value: int64(1),
				},
			},
		},
		Inputs: map[string]*framework.TensorProto{
			"map/TensorArrayStack/TensorArrayGatherV3": &framework.TensorProto{
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

	/*
			file, err := os.Open(imgPath)
			if err != nil {
				log.Fatal(err)
				os.Exit(1)
			}
			defer file.Close()

			decodedImg, _, err := image.Decode(file)
			if err != nil {
				log.Fatal(err)
				os.Exit(1)
			}

			resizedImg := resize.Resize(224, 224, decodedImg, resize.Lanczos3)

			// convert to NCHW
			imgSlice := imageToSlice(resizedImg, true, true)
			flattenImgSlice := flattenSlice(imgSlice, []uint32{1, 3, 224, 224})

			// Get the slice header
			header := *(*reflect.SliceHeader)(unsafe.Pointer(&flattenImgSlice))

			const SIZEOF_FP32 = 4 // bytes
			header.Len *= SIZEOF_FP32
			header.Cap *= SIZEOF_FP32

			// Convert slice header to an []int32
			byte_data := *(*[]byte)(unsafe.Pointer(&header))

			predictRequest := newPredictRequest("resnet", 1)
			predictRequest.ModelSpec.SignatureName = "serving_default"

		err = addInput(predictRequest, "map/TensorArrayStack/TensorArrayGatherV3", framework.DataType_DT_FLOAT, byte_data, []int64{1, 3, 224, 224})
		if err != nil {
			fmt.Printf("Error adding input tensor: %v:", err)
		}
	*/
	conn, err := grpc.Dial(*servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Cannot connect to the grpc server: %v\n", err)
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)

	predictResponse, err := client.Predict(context.Background(), predictRequest)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println("Request sent successfully")

	tp := predictResponse.Outputs["softmax_tensor"]
	responseContent := tp.GetTensorContent()

	outputShape := tp.GetTensorShape()
	dim := outputShape.GetDim()

	classesNum := dim[1].GetSize()

	outMat, err := gocv.NewMatFromBytes(1, int(classesNum), gocv.MatTypeCV32FC1, responseContent)
	outMat = outMat.Reshape(1, 1)

	_, maxVal, _, maxLoc := gocv.MinMaxLoc(outMat)

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
