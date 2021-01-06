/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  tensorflow::string callPredict(const tensorflow::string& model_name,
                                 const tensorflow::string& model_signature_name,
                                 const tensorflow::string& file_path) {
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point opencv2_time;
    std::chrono::high_resolution_clock::time_point serialize_time;
    std::chrono::high_resolution_clock::time_point predict_time;
    std::chrono::high_resolution_clock::time_point postprocess_time;
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);
    predictRequest.mutable_model_spec()->set_signature_name(
        model_signature_name);

    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
        *predictRequest.mutable_inputs();

    tensorflow::TensorProto proto;

    cv::Mat image;
    image = cv::imread( file_path, 1 );
    cv::Mat image224;
    cv::Size size(224, 224);
    cv::resize(image, image224, size);
    cv::Mat image224_32;
    image224.convertTo(image224_32, CV_32F);
    opencv2_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::to_string(std::chrono::duration<double, std::milli>(opencv2_time - start_time).count());
    std::cout << "Image imported " << file_path << "in" << elapsed << "ms" << std::endl;;
    start_time = std::chrono::high_resolution_clock::now();
    inputs["0"] = serializeImageToTensorProto(&image224_32);
    serialize_time = std::chrono::high_resolution_clock::now();
    elapsed = std::to_string(std::chrono::duration<double, std::milli>(serialize_time - start_time).count());
    std::cout << "request serialized in" << elapsed << "ms" << std::endl;;
    start_time = std::chrono::high_resolution_clock::now();
    Status status = stub_->Predict(&context, predictRequest, &response);
    predict_time = std::chrono::high_resolution_clock::now();
    elapsed = std::to_string(std::chrono::duration<double, std::milli>(predict_time - start_time).count());
    std::cout << "Prediction received in " << elapsed << "ms" << std::endl;;

    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is " << response.outputs_size() << std::endl;
      OutMap& map_outputs = *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;
      start_time = std::chrono::high_resolution_clock::now();
      for (iter = map_outputs.begin(); iter != map_outputs.end(); ++iter) {
        tensorflow::TensorProto& result_tensor_proto = iter->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {

          // alternative to deserialize response without Tensor objects
          // auto results_vector = const_cast<float*>(reinterpret_cast<const float*>(result_tensor_proto.tensor_content().data()));
          // std::cout << "test value" << test[0];
          auto output_tensor = tensor.tensor<float, 2>();
          int index{0};
          float max_value{-1000};
          int i = 0;
          for(i=0; i < 1000; i++){
              if (max_value < output_tensor(0,i)){
                  index = i;
                  max_value = output_tensor(0,i);
              }
          }
          postprocess_time = std::chrono::high_resolution_clock::now();
          elapsed = std::to_string(std::chrono::duration<double, std::milli>(postprocess_time - start_time).count());
          std::cout << "Response postprocessing in " << elapsed << "ms" << std::endl;;
          std::cout << "max class" << index << " max value:"<< max_value << std::endl;


          std::cout << "the result tensor[" << output_index
                    << "] is:" << std::endl
                    << tensor.SummarizeValue(10) << std::endl;
          std::cout << "Shape [" << tensor.shape().dim_size(0) << "," << tensor.shape().dim_size(1) << "]" << std::endl;



        } else {
          std::cout << "the result tensor[" << output_index
                    << "] convert failed." << std::endl;
        }
        ++output_index;
      }
      return "Done.";
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
      return "gRPC failed.";
    }
  }
    auto reorder_to_chw(cv::Mat* mat) {
        assert(mat->channels() == 3);
        std::vector<float> data(mat->channels() * mat->rows * mat->cols);
        for(int y = 0; y < mat->rows; ++y) {
            for(int x = 0; x < mat->cols; ++x) {
                for(int c = 0; c < mat->channels(); ++c) {
                    data[c * (mat->rows * mat->cols) + y * mat->cols + x] = mat->at<cv::Vec3f>(y, x)[c];
                }
            }
        }
    return data;
    }

    tensorflow::TensorProto serializeImageToTensorProto(cv::Mat* image) {
      tensorflow::TensorProto proto;
      proto.Clear();
      proto.set_dtype(tensorflow::DataTypeToEnum<float>::value);

      proto.mutable_tensor_shape()->Clear();
      proto.mutable_tensor_shape()->add_dim()->set_size(1);
      proto.mutable_tensor_shape()->add_dim()->set_size(image->channels());
      proto.mutable_tensor_shape()->add_dim()->set_size(image->rows);
      proto.mutable_tensor_shape()->add_dim()->set_size(image->cols);
      auto image_nchw = reorder_to_chw(image);
      proto.mutable_tensor_content()->assign((char*)image_nchw.data(), image->total() * image->elemSize());
      return proto;
    }


 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char** argv) {
  tensorflow::string server_port = "localhost:8500";
  tensorflow::string image_file = "";
  tensorflow::string model_name = "resnet";
  tensorflow::string model_signature_name = "serving_default";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port,
                       "the IP and port of the server"),
      tensorflow::Flag("image_file", &image_file, "the path to the image"),
      tensorflow::Flag("model_name", &model_name, "name of model"),
      tensorflow::Flag("model_signature_name", &model_signature_name,
                       "name of model signature")};

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || image_file.empty()) {
    std::cout << usage;
    return -1;
  }

  ServingClient guide(
      grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
  std::cout << "calling predict using file: " << image_file << "  ..."
            << std::endl;
  for (int n=10; n>0; n--) {
    std::cout << guide.callPredict(model_name, model_signature_name, image_file)
            << std::endl;
  }
  return 0;
}
