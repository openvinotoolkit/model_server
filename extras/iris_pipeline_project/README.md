# Optimized serving endpoint for Classical Machine Learning Models 

This project presents a versatile framework for training and deploying classical machine learning models on the OpenVINO Model Server (OVMS), leveraging accelerators(oneDAL for scikit-learn and IPEX for pytorch algorithms) to boost inference performance. The architecture is designed as a general solution, accommodating various machine learning algorithms and providing a consistent structure for both model development and production deployment.

At the moment, for demonstrating purposes of the established project structure, we'll be demonstrating the Logistic Regression Model in pytorch and KMeans algorithm using scikit-Learn .

---

## Instructions for preparing the data
Run the command to download the Iris dataset, which is taken to be the hello-world dataset of classification datasets for simplicity purposes.
Split and preprocess(label encoding, normalization and Cross validation) accordingly.

```bash
curl -o iris.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/extras/iris_pipeline_project
```
---


## Step 2: Build and Run OVMS Docker Image

### 2.1. Build the Docker Image

```bash
docker build --no-cache -t prototype_iris .
```

### 2.2. Run the OVMS Container

```bash
 docker run --rm -it -v $(pwd):/workspace -p 9000:9000 prototype_iris --config_path /workspace/model_config.json --port 9000 --log_level DEBUG
```
- **Note:** Adjust `$(pwd)` if you are running from a different working directory.

---

##  Step 3: Project Structure

```
client/
  ├── client_inference.py
  └── client_train.py
pipeline/
  ├── graph.pbtxt
  ├── model.py
  └── ovmsmodel.py
Dockerfile
model_config.json
kmeans_params.json
hyperparams.json

```

---

## Step 4: Run Training and Inference

### 4.1. Training

```bash
python client/client_train.py train iris_train.csv Species --params hyperparams.json --encode Species --model_class LogisticRegressionTorch

python client/client_train.py train iris_train.csv Species --params kmeans_params.json --encode Species --model_class KMeansSkLearn


```

### 4.2. Inference

```bash
python client/client_inference.py infer data_folder/iris_test.csv --target_column Species  --model_class LogisticRegressionTorch

python client/client_inference.py infer iris_train_nolabel.csv  --target_column Species --model_class KMeansSkLearn

```

---

For Enabling accelerator support:
You can pass the ```bool - (use_ipex/use_oneDAL)``` in the "hyperparams.json"/ "kmeans_params.json" file as a key variable pair to either True/False depending on the necessity.


Command-Line Usage:

The training and inference client supports flexible options for both Logistic Regression and KMeans models.

Usage
python client/client_train.py <train|infer> <path_to_csv> <target_column (or NONE for KMeans)> \
    [--params <path_to_params_json>] [--encode <col1,col2,...>] [--model_class <ModelClassName>]

Arguments

    train|infer
    Mode of operation.

        train: Train a new model with the provided dataset.

        infer: Run inference using a trained model.

    <path_to_csv>
    Path to the dataset in CSV format.

    <target_column>

        For classification (Logistic Regression): name of the target column.

        For clustering (KMeans): use NONE.

    --params <path_to_params_json> (optional)
    Path to a JSON file containing model hyperparameters.
    Example:

{
  "max_iter": 300,
  "solver": "lbfgs",
  "random_state": 42,
  "n_clusters": 3
}

    If not provided, default parameters are used.

--encode <col1,col2,...> (optional)
Comma-separated list of categorical column names to encode.

    Encoding can also be performed client-side before sending data to the server.

    If omitted, no encoding is applied.

--model_class <ModelClassName> (optional)
Specify the model class explicitly (e.g., LogisticRegression, KMeans).

    Defaults are inferred from the mode and target column.

---

## Troubleshooting

- **Logs:**  
  For debugging, check OVMS container logs:
  ```bash
  docker logs prototype_iris
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
Model trained successfully

```
For Inference:

```
Read CSV file successfully
Inference mode detected.
Inference predictions: [...]

```

---

NOTE: Cluster assignments and centroid details are available in the container logs. Since the terminal is non-GUI, .show() visualization is not supported.
