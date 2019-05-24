# OpenVINO container for kubeflow pipeline

This docker container includes OpenVINO infernece engine, model optimizer and Tensorflow tools to generate 
exemplary image classification models.

It contains three scripts:

- [slim_model.py](tf_slim.md) - generates TensorFlow slim models based on pretrained checkpoints
- [convert_model.py](mo.md) - optimizes models and saves them in Intermediate Representation format
- [predict.py](predict.md) - runs the model evaluation


## Docker image building

```bash
docker build -t openvino_pipeline --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy .
```
