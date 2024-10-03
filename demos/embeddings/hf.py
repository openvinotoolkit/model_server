from transformers import AutoTokenizer, AutoModel
import torch
import datetime

# Sentences we want sentence embeddings for
sentences = ["Model Server hosts models and makes them accessible to software components over standard network protocols: a client sends a request to the model server, which performs model inference and sends a response back to the client. Model Server offers many advantages for efficient model deployment",
              "OpenVINO™ Model Server (OVMS) is a high-performance system for serving models. Implemented in C++ for scalability and optimized for deployment on Intel architectures, the model server uses the same architecture and API as TensorFlow Serving and KServe while applying OpenVINO for inference execution. Inference service is provided via gRPC or REST API, making deploying new algorithms and AI experiments easy.",
              "Inference Optimization: Boost deep learning performance in computer vision, automatic speech recognition, generative AI, natural language processing with large and small language models, and many other common tasks.",
              "Flexible Model Support: Use models trained with popular frameworks such as TensorFlow, PyTorch, ONNX, Keras, and PaddlePaddle. Convert and deploy models without original frameworks."]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
model.eval()

# Tokenize sentences
warmup_input = tokenizer(["my test", "test test"], padding=True, truncation=True, return_tensors='pt')

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print('input_ids', encoded_input.input_ids.shape)
print('token_type_ids', encoded_input.token_type_ids.shape)
print('attention_mask', encoded_input.attention_mask.shape)
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**warmup_input)
    start_time = datetime.datetime.now()
    model_output = model(**encoded_input)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    print("Duration:", duration, "ms")
    # Perform pooling. In this case, cls pooling.
    print('last_hidden_state', model_output.last_hidden_state.shape)
    print('pooler_output', model_output.pooler_output.shape)
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
#sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
#print("Sentence embeddings:", sentence_embeddings)