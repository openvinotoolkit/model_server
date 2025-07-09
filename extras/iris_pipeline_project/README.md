# OVMS Iris Pipeline Example

This repository demonstrates how to use OpenVINO Model Server (OVMS) with a custom Mediapipe pipeline for the Iris dataset, including both model training and inference through a Python client.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd [need to fix]
```
---


## Step 2: Build and Run OVMS Docker Image

### 2.1. Build the Docker Image

```bash
docker build --no-cache -t iris_7 .
```

### 2.2. Run the OVMS Container

```bash
docker run --rm -it -v "$PWD:/workspace"   -p 9000:9000 -p 8000:8000  iris_7 --config_path /workspace/model_config.json   --port 9000 --rest_port 8000
```
- **Note:** Adjust `$(pwd)` if you are running from a different working directory.

---

##  Step 3: Project Structure

```
client/
  ├── client_inference.py
  └── client_train.py
pipeline/
  ├── __pycache__/
  ├── graph.pbtxt
  └── ovmsmodel.py
Dockerfile
model_config.json
```

---

## Step 4: Run Training and Inference

### 4.1. Training

```bash
python client/client_train.py train data/iris_train.csv
```

### 4.2. Inference

```bash
python client/client_inference.py infer data/iris_test.csv
```

---

## Input Format

The pipeline expects input as a JSON object, sent as a single-element numpy array of bytes (`dtype=object`):

```json
{
  "mode": "train" | "infer",
  "data": "<CSV string>"
}
```

---

## Troubleshooting

- **Logs:**  
  For debugging, check OVMS container logs:
  ```bash
  docker logs iris_7
  ```
- **Code Changes:**  
  After editing `pipeline/ovmsmodel.py`, **restart the OVMS container** for changes to take effect.

- **If nothing prints from your Python node:**  
  - Use `flush=True` in your print statements.
  - Print to `sys.stderr`.
  - Try writing to a file inside the container for debug.

---

## Example Output
For Training:

```
Read CSV file successfully
Training mode detected. Preparing data for training...
Connected to OVMS at localhost:9000
Server response decoded: string -  [...]
The output string formatted as: [<1 - Model trained successfully | 0 - Otherwise>   <Accuracy>  <Precision>  <Recall>  <f1-score>]
```
For Inference:

```
Read CSV file successfully
Inference mode detected.
Inference predictions: [...]

```

---

## 📄 License

MIT License
