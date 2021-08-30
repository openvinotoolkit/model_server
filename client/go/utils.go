package main

import (
	"image"
	_ "image/jpeg" // need to make sure the right codes are included
	_ "image/png"  // need to make sure the right codes are included

	"errors"
	"fmt"
	"reflect"

	framework "tensorflow/core/framework"
	pb "tensorflow_serving"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
)

func newPredictRequest(modelName string, modelVersion int64) (pr *pb.PredictRequest) {
	return &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name: modelName,
			VersionChoice: &pb.ModelSpec_Version{
				Version: &google_protobuf.Int64Value{
					Value: modelVersion,
				},
			},
		},
		Inputs: make(map[string]*framework.TensorProto),
	}
}

// if tensor is one dim, shapeSize is nil
func addInput(predictRequest *pb.PredictRequest, inputName string, dataType framework.DataType, tensor interface{},
	shapeSize []int64) (err error) {
	values := reflect.ValueOf(tensor)
	if values.Kind() != reflect.Slice {
		return errors.New("tensor must be slice")
	}
	inputSize := values.Len()
	fmt.Printf("inputSize: %d\n", inputSize)

	tensorProto := &framework.TensorProto{
		Dtype: dataType,
	}

	var ok bool
	tensorProto.TensorContent, ok = tensor.([]byte)

	if !ok {
		if err != nil {
			err = errors.New("Type switch failed")
		}
		fmt.Println("returning after type switch failed")
		return
	}

	fmt.Println("inputName: ", inputName)

	tensorProto.TensorShape = &framework.TensorShapeProto{
		Dim: []*framework.TensorShapeProto_Dim{},
	}

	for _, size := range shapeSize {
		name := ""
		tensorProto.TensorShape.Dim = append(tensorProto.TensorShape.Dim, &framework.TensorShapeProto_Dim{
			Size: size,
			Name: name,
		})
	}

	predictRequest.Inputs[inputName] = tensorProto
	return
}

func flattenSlice(src [][][]uint32, shape []uint32) (dst []uint32) {
	// get total size
	// TODO use passed structure instead
	fmt.Printf("imgResized size: [%d, %d, %d]\n", len(src), len(src[0]), len(src[0][0]))
	var size uint32 = 1
	for i := range shape {
		size *= shape[i]
	}

	fmt.Printf("flattenImg expected size [num of elements]: %d\n", size)
	dst = make([]uint32, 0, size)

	// Now assemble the slices
	for i1 := range src {
		for i2 := range src[i1] {
			dst = append(dst, src[i1][i2]...)
		}
	}

	fmt.Printf("flattenImg actual size [num of elements]: %d\n", len(dst))
	return
}

func imageToSlice(src image.Image, nchw bool, bgr bool) (dst [][][]uint32) {
	bounds := src.Bounds()
	if nchw {
		dst = make([][][]uint32, 3) // channels

		for channel := range dst {
			dst[channel] = make([][]uint32, (bounds.Max.Y - bounds.Min.Y)) // rows

			for row := range dst[channel] {
				dst[channel][row] = make([]uint32, (bounds.Max.X - bounds.Min.X)) // cols
			}
		}

		for row := range dst[0] {
			for col := range dst[0][row] {
				r, g, b, _ := src.At(bounds.Min.X+col, bounds.Min.Y+row).RGBA()

				if bgr {
					dst[0][row][col] = b / 255
					dst[1][row][col] = g / 255
					dst[2][row][col] = r / 255
				} else {
					dst[0][row][col] = r / 255
					dst[1][row][col] = g / 255
					dst[2][row][col] = b / 255
				}
			}
		}

	} else {
		dst = make([][][]uint32, (bounds.Max.Y - bounds.Min.Y)) // rows

		for row := range dst {
			dst[row] = make([][]uint32, (bounds.Max.X - bounds.Min.X)) // cols

			for col := range dst[row] {
				dst[row][col] = make([]uint32, 3) // channels
				r, g, b, _ := src.At(bounds.Min.X+col, bounds.Min.Y+row).RGBA()

				if bgr {
					dst[row][col][0] = b / 255
					dst[row][col][1] = g / 255
					dst[row][col][2] = r / 255
				} else {
					dst[row][col][0] = r / 255
					dst[row][col][1] = g / 255
					dst[row][col][2] = b / 255
				}
			}
		}
	}
	return
}
