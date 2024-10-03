import numpy as np
import tritonclient.grpc as grpcclient
import datetime

client = grpcclient.InferenceServerClient("localhost:11337")
sentences = ["Model Server hosts models and makes them accessible to software components over standard network protocols: a client sends a request to the model server, which performs model inference and sends a response back to the client. Model Server offers many advantages for efficient model deployment",
              "OpenVINO™ Model Server (OVMS) is a high-performance system for serving models. Implemented in C++ for scalability and optimized for deployment on Intel architectures, the model server uses the same architecture and API as TensorFlow Serving and KServe while applying OpenVINO for inference execution. Inference service is provided via gRPC or REST API, making deploying new algorithms and AI experiments easy.",
              "Inference Optimization: Boost deep learning performance in computer vision, automatic speech recognition, generative AI, natural language processing with large and small language models, and many other common tasks.",
              "Flexible Model Support: Use models trained with popular frameworks such as TensorFlow, PyTorch, ONNX, Keras, and PaddlePaddle. Convert and deploy models without original frameworks."]

input = np.array(sentences, dtype=np.object_)
infer_input = grpcclient.InferInput("string_input", [4], "BYTES")
#infer_input = grpcclient.InferInput("Parameter_1", [4], "BYTES")
#infer_input = grpcclient.InferInput("string_input", [4], "BYTES")   # for rerank model
infer_input.set_data_from_numpy(input)
start_time = datetime.datetime.now()
results = client.infer("test", [infer_input])
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
print("Duration:", duration, "ms")
out = results.as_numpy("last_hidden_state")
print(out)
print(out.shape)