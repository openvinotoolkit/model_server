#include <iostream>
#include <openvino/openvino.hpp>


void infer_dummy_with_elem_num(int elem_num, ov::runtime::InferRequest& infer_request) {
    ov::runtime::Tensor tensor(ov::element::f32, {1, elem_num});
    float* data = tensor.data<float>();

    for (int i = 0; i < elem_num; i++) {
        data[i] = i + 1.2f;
        std::cout << data[i] << ",";
    }
    std::cout << std::endl;

    infer_request.set_input_tensor(tensor);
    infer_request.infer();

    ov::runtime::Tensor output = infer_request.get_output_tensor();

    float* output_data = output.data<float>();
    for (int i = 0; i < elem_num; i++) {
        std::cout << output_data[i] << ",";
    }
    std::cout << std::endl;
}

void infer_resnet_with_resolution(int resolution, ov::runtime::InferRequest& infer_request) {
    ov::runtime::Tensor tensor(ov::element::f32, {1, 3, resolution, resolution});
    float* data = tensor.data<float>();

    for (int i = 0; i < resolution * resolution * 3; i++) {
        data[i] = i + 1.2f;
    }

    infer_request.set_input_tensor(tensor);
    infer_request.infer();

    ov::runtime::Tensor output = infer_request.get_output_tensor();

    float* output_data = output.data<float>();
    for (int i = 0; i < 10; i++) {
        std::cout << output_data[i] << ",";
    }
    std::cout << "...";
    for (int i = (resolution * resolution * 3 - 1) - 10; i < (resolution * resolution * 3); i++) {
        std::cout << output_data[i] << ",";
    }
    std::cout << std::endl;
}

int main() {
    ov::runtime::Core ie;
    //std::shared_ptr<ov::Function> model = ie.read_model("src/test/dummy/1/dummy.xml");
    //std::shared_ptr<ov::Function> model = ie.read_model("/workspace/models/resnet50-binary/1/resnet50-binary-0001.xml");
    std::shared_ptr<ov::Function> model = ie.read_model("/workspace/models/bert-base-chinese-xnli-zh-fp32-onnx-0001/1/bert-base-chinese-xnli-zh-fp32-onnx-0001.xml");

    //ov::PartialShape input_shape{1, ngraph::Dimension(1, 50)};
    //std::map<std::string, ov::PartialShape> new_shapes{{"b", input_shape}};
    //model->reshape(new_shapes);

    // ov::PartialShape input_shape{1, 3, ngraph::Dimension(220, 360), ngraph::Dimension(220, 360)};
    // std::map<std::string, ov::PartialShape> new_shapes{{"0", input_shape}};
    // model->reshape(new_shapes);

    ov::PartialShape input_shape{1, ngraph::Dimension(32, 256)};
    std::map<std::string, ov::PartialShape> new_shapes{
        {"0", input_shape},
        {"1", input_shape},
        {"2", input_shape},
    };
    model->reshape(new_shapes);
    std::cout << "1" << std::endl;
    ov::runtime::ExecutableNetwork exec_network = ie.compile_model(model, "CPU");
    ov::runtime::InferRequest infer_request = exec_network.create_infer_request();

    ov::runtime::Tensor tensor0(ov::element::i32, {1, 100});
    ov::runtime::Tensor tensor1(ov::element::i32, {1, 100});
    ov::runtime::Tensor tensor2(ov::element::i32, {1, 100});

    infer_request.set_tensor("0", tensor0);
    infer_request.set_tensor("1", tensor1);
    infer_request.set_tensor("2", tensor2);
    std::cout << "2" << std::endl;

    infer_request.infer();
    std::cout << "3" << std::endl;



    // infer_dummy_with_elem_num(1, infer_request);
    // infer_dummy_with_elem_num(2, infer_request);
    // infer_dummy_with_elem_num(14, infer_request);

    // infer_resnet_with_resolution(224, infer_request);
    // infer_resnet_with_resolution(256, infer_request);

    return 0;
}
