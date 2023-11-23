import tritonclient.grpc as grpcclient
import threading
import time
from PIL import Image
from io import BytesIO

client = grpcclient.InferenceServerClient("localhost:11339")
data = "Zebras in space".encode()

model_name = "python_model"
input_name = "text"

start = time.time()
infer_input = grpcclient.InferInput(input_name, [len(data)], "BYTES")
infer_input._raw_content = data

results = client.infer(model_name, [infer_input])
img = Image.open(BytesIO(results.as_numpy("OUTPUT")))
img.save(f"output.png")
duration = time.time() - start
print(f"Total workers time: {duration} s")
