import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:11339")
text = "He never went out without a book under his arm, and he often came back with two."
print(f"Text:\n{text}\n")
data = text.encode()
infer_input = grpcclient.InferInput("text", [len(data)], "BYTES")
infer_input._raw_content = data
results = client.infer("python_model", [infer_input])
print(f"Translation:\n{results.as_numpy('OUTPUT').tobytes().decode()}\n")
