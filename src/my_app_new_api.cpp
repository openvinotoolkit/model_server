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
    infer_request.start_async();
    infer_request.wait();

    ov::runtime::Tensor output = infer_request.get_output_tensor();

    float* output_data = output.data<float>();
    for (int i = 0; i < elem_num; i++) {
        std::cout << output_data[i] << ",";
    }
    std::cout << std::endl;
}

int argmax(ov::runtime::Tensor& output) {
    int label = -1;
    float val = 0;

    for (size_t i = 0; i < output.get_byte_size() / output.get_element_type().size(); i++) {
        float v = output.data<float>()[i];
        if (label == -1 || v > val) {
            val = v;
            label = i;
        }
    }
    return label;
}

void infer_resnet_with_resolution(int resolution, ov::runtime::InferRequest& infer_request) {
    ov::runtime::Tensor tensor(ov::element::f32, {1, 3, resolution, resolution});
    float* data = tensor.data<float>();

    for (int i = 0; i < resolution * resolution * 3; i++) {
        data[i] = i + 1.2f;
    }

    infer_request.set_input_tensor(tensor);
    infer_request.start_async();
    infer_request.wait();

    ov::runtime::Tensor output = infer_request.get_output_tensor();

    float* output_data = output.data<float>();
    for (int i = 0; i < 10; i++) {
        std::cout << output_data[i] << ",";
    }
    std::cout << "...";
    for (int i = (output.get_byte_size() / output.get_element_type().size() - 1) - 10; i < output.get_byte_size() / output.get_element_type().size(); i++) {
        std::cout << output_data[i] << ",";
    }
    std::cout << std::endl;
    std::cout << "label: " << argmax(output) << std::endl;
}

void infer_bert_with_size(int size, ov::runtime::InferRequest& infer_request) {
    ov::runtime::Tensor tensor0(ov::element::i32, {1, size});
    ov::runtime::Tensor tensor1(ov::element::i32, {1, size});
    ov::runtime::Tensor tensor2(ov::element::i32, {1, size});
    infer_request.set_tensor("0", tensor0);
    infer_request.set_tensor("1", tensor1);
    infer_request.set_tensor("2", tensor2);
    std::cout << "2" << std::endl;
    infer_request.start_async();
    infer_request.wait();
    std::cout << "3" << std::endl;
}

int main() {
    ov::runtime::Core ie;

    //std::shared_ptr<ov::Function> model = ie.read_model("src/test/dummy/1/dummy.xml");
    std::shared_ptr<ov::Function> model = ie.read_model("/workspace/models/resnet50-binary/1/resnet50-binary-0001.xml");
    //std::shared_ptr<ov::Function> model = ie.read_model("/workspace/models/bert-base-chinese-xnli-zh-fp32-onnx-0001/1/bert-base-chinese-xnli-zh-fp32-onnx-0001.xml");

    //ov::Output<ov::Node> input = model->input("1");
    //ov::Shape shape = input.get_shape();
    //ov::element::Type type = input.get_element_type();
    //std::cout << "Input: 1; Shape: " << shape << "; Type: " << type << std::endl;

    // ov::PartialShape input_shape{1, ngraph::Dimension(1, 50)};
    // std::map<std::string, ov::PartialShape> new_shapes{{"b", input_shape}};
    // model->reshape(new_shapes);

    // ov::PartialShape input_shape{1, 3, ngraph::Dimension(220, 360), ngraph::Dimension(220, 360)};
    // std::map<std::string, ov::PartialShape> new_shapes{{"0", input_shape}};
    // model->reshape(new_shapes);

    //ov::PartialShape input_shape{1, ngraph::Dimension(32, 256)};
    //std::map<std::string, ov::PartialShape> new_shapes{
    //    {"0", input_shape},
    //    {"1", input_shape},
    //    {"2", input_shape},
    //};
    //model->reshape(new_shapes);
    //std::cout << "1" << std::endl;
    ov::runtime::ExecutableNetwork exec_network = ie.compile_model(model, "CPU");

    ov::runtime::InferRequest infer_request = exec_network.create_infer_request();

    // infer_dummy_with_elem_num(1, infer_request);
    // infer_dummy_with_elem_num(2, infer_request);
    // infer_dummy_with_elem_num(14, infer_request);

    infer_resnet_with_resolution(224, infer_request);
    // infer_resnet_with_resolution(256, infer_request);

    //infer_bert_with_size(128, infer_request);
    //infer_bert_with_size(128, infer_request);
    //infer_bert_with_size(128, infer_request);
    //infer_bert_with_size(100, infer_request);

    return 0;
}
