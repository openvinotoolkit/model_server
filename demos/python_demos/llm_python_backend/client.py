import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:11339")
text = "Describe the state of the healthcare industry in the United States in max 2 sentences"
print(f"Question:\n{text}\n")
data = text.encode()
infer_input = grpcclient.InferInput("pre_prompt", [len(data)], "BYTES")
infer_input._raw_content = data
results = client.infer("python_model", [infer_input], timeout=0)
print(f"Completion:\n{results.as_numpy('OUTPUT').tobytes().decode()}\n")
