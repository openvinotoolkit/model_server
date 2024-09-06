# Embeddings benchmark

## Setting up embedding serving
```bash
 docker run -p 7997:7997 michaelf34/infinity:latest v2 --model-id BAAI/bge-small-en-v1.5
```

## Preparing to run benchmark

```bash
pip install -r requirements.txt
```

## Running benchmark

```bash
python3 demo.py
Elapsed time:  41.75693106651306 s
```
