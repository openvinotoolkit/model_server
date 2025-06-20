import numpy as np
import tritonclient.grpc as grpcclient
import pandas as pd


X_test = pd.read_csv("/home/harshitha/iris_pipeline_project/data/iris_test.csv")
X_test = X_test.values.astype(np.float32)  

input_name = "input"   
output_name = "output_label" 

inputs = []
infer_input = grpcclient.InferInput(input_name, X_test.shape, "FP32")
infer_input.set_data_from_numpy(X_test)
inputs.append(infer_input)

outputs = [grpcclient.InferRequestedOutput(output_name)]

client = grpcclient.InferenceServerClient(url="localhost:9000")

model_name = "iris_logreg"
response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

predictions = response.as_numpy(output_name)
print("Predictions:", predictions)