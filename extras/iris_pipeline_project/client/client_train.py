import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient

data = pd.read_csv("/home/harshitha/iris_pipeline_project/data/iris_train.csv")
df = pd.DataFrame(data)

csv_str = df.to_csv(index=False)
csv_bytes = np.frombuffer(csv_str.encode('utf-8'), dtype=np.uint8)

input_name = "input"
input = []
infer_input = grpcclient.InferInput(input_name, csv_bytes.shape, "UINT8")
infer_input.set_data_from_numpy(csv_bytes)
input.append(infer_input)

output_name = "output_label"
output_label = [grpcclient.InferRequestedOutput(output_name)]

client = grpcclient.InferenceServerClient(url="localhost:9000")

model_name = "iris_pipeline"
response = client.infer(model_name=model_name, inputs=input, outputs=output_label)

result = response.as_numpy(output_name)
print("Training result:", result)