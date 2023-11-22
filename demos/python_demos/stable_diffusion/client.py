import tritonclient.grpc as grpcclient
import threading
import time
from PIL import Image
from io import BytesIO

client = grpcclient.InferenceServerClient("localhost:11339")
data = "Zebra in space".encode()

def run_inference(model_name, input_name, iterations):
    infer_input = grpcclient.InferInput(input_name, [len(data)], "BYTES")
    infer_input._raw_content = data

    for _ in range(iterations):
        results = client.infer(model_name, [infer_input])
    img = Image.open(BytesIO(results.as_numpy("OUTPUT")))
    img.save("output.png")

configs = [(1,1)]

for num_threads, iterations in configs:
    print(f"\nNumber of threads: {num_threads}, Iterations per thread: {iterations}")
    print("Running with OpenVINO backend")
    model_name = "python_model"
    input_name = "text"
    threads = [threading.Thread(target=run_inference, args=[model_name, input_name, iterations]) for _ in range(num_threads)]
    for thread in threads:
        thread.start()
    start = time.time()
    for thread in threads:
        thread.join()
    duration = time.time() - start
    print(f"Total workers time: {duration} s")
    print(f"Estimated FPS: {(num_threads * iterations) / duration}")