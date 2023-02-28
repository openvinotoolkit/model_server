# Measure performance
## Prerequisites

Clone OVMS repository.
```bash
$ git clone https://github.com/openvinotoolkit/model_server.git
$ cd model_server/tests/performance
```

Enable virtualenv in project root directory, install requirements.txt.
```bash
$ virtualenv .venv
$ . .venv/bin/activate
$ pip3 install -r ../requirements.txt
```

```bash
$ wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
$ docker run -u $(id -u) -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name resnet \
--model_path /models/resnet50 --port 9178 --batch_size 2
```

## Latency
```bash
$ python3 grpc_latency.py --help
usage: grpc_latency.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
                       [--labels_numpy_path LABELS_NUMPY_PATH]
                       [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]
                       [--input_name INPUT_NAME] [--output_name OUTPUT_NAME]
                       [--iterations ITERATIONS] [--batchsize BATCHSIZE]
                       [--model_name MODEL_NAME]
                       [--model_version MODEL_VERSION]
                       [--report_every REPORT_EVERY] [--precision PRECISION]
                       [--id ID]

Sends requests via TFS gRPC API using images in numpy format. It measures
performance statistics.

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        image in numpy format
  --labels_numpy_path LABELS_NUMPY_PATH
                        labels in numpy format
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9178
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output tensor name. default: prob
  --iterations ITERATIONS
                        Number of requests iterations, as default use number
                        of images in numpy memmap. default: 0 (consume all
                        frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name in payload. default: resnet
  --model_version MODEL_VERSION
                        Model version number. default: 1
  --report_every REPORT_EVERY
                        Report performance every X iterations
  --precision PRECISION
                        input precision
  --id ID               Helps identifying client

```

### Example usage:
```bash
$ python3 grpc_latency.py --grpc_address localhost --grpc_port 9000 --images_numpy_path imgs.npy --iteration 1000 --batchsize 2 --report_every 100 --input_name "0"
```
```bash
[--] Starting iterations
[--] Iteration   100/ 1000; Current latency: 1.45ms; Average latency: 3.78ms
[--] Iteration   200/ 1000; Current latency: 1.77ms; Average latency: 2.63ms
[--] Iteration   300/ 1000; Current latency: 1.62ms; Average latency: 2.29ms
[--] Iteration   400/ 1000; Current latency: 1.63ms; Average latency: 2.07ms
[--] Iteration   500/ 1000; Current latency: 1.36ms; Average latency: 1.97ms
[--] Iteration   600/ 1000; Current latency: 1.38ms; Average latency: 1.88ms
[--] Iteration   700/ 1000; Current latency: 1.33ms; Average latency: 1.81ms
[--] Iteration   800/ 1000; Current latency: 1.50ms; Average latency: 1.75ms
[--] Iteration   900/ 1000; Current latency: 1.66ms; Average latency: 1.74ms
[--] Iterations:  1000; Final average latency: 1.73ms
```

## Throughput
Script `grpc_throughput.sh 28` spawns 28 gRPC clients.

### Example usage:
```bash
$ ./grpc_throughput.sh 28 --grpc_address localhost --grpc_port 9000 --images_numpy_path imgs.npy --iteration 4000 --batchsize 2 --input_name "0"
```

This will create `28 * 4000` requests, `2` frames each, = `224000` frames total.  
To get frames per second, script outputs total execution time:

```bash
[3 ] Starting iterations
[17] Starting iterations
[2 ] Starting iterations
[21] Starting iterations
[4 ] Starting iterations
[10] Starting iterations
[7 ] Starting iterations
[20] Starting iterations
[23] Starting iterations
[28] Starting iterations
[25] Starting iterations
[26] Starting iterations
[13] Starting iterations
[18] Starting iterations
[19] Starting iterations
[6 ] Starting iterations
[15] Starting iterations
[24] Starting iterations
[5 ] Starting iterations
[14] Starting iterations
[11] Starting iterations
[16] Starting iterations
[12] Starting iterations
[22] Starting iterations
[9 ] Starting iterations
[8 ] Starting iterations
[1 ] Starting iterations
[27] Starting iterations
[3 ] Iterations:  4000; Final average latency: 11.67ms
[11] Iterations:  4000; Final average latency: 12.04ms
[4 ] Iterations:  4000; Final average latency: 12.57ms
[16] Iterations:  4000; Final average latency: 12.54ms
[12] Iterations:  4000; Final average latency: 12.31ms
[28] Iterations:  4000; Final average latency: 12.71ms
[8 ] Iterations:  4000; Final average latency: 12.72ms
[1 ] Iterations:  4000; Final average latency: 12.63ms
[13] Iterations:  4000; Final average latency: 12.33ms
[5 ] Iterations:  4000; Final average latency: 12.76ms
[6 ] Iterations:  4000; Final average latency: 12.53ms
[9 ] Iterations:  4000; Final average latency: 13.21ms
[2 ] Iterations:  4000; Final average latency: 12.90ms
[18] Iterations:  4000; Final average latency: 12.83ms
[21] Iterations:  4000; Final average latency: 13.23ms
[25] Iterations:  4000; Final average latency: 12.74ms
[23] Iterations:  4000; Final average latency: 13.28ms
[27] Iterations:  4000; Final average latency: 13.28ms
[26] Iterations:  4000; Final average latency: 12.77ms
[20] Iterations:  4000; Final average latency: 13.07ms
[24] Iterations:  4000; Final average latency: 13.01ms
[15] Iterations:  4000; Final average latency: 12.91ms
[10] Iterations:  4000; Final average latency: 12.88ms
[14] Iterations:  4000; Final average latency: 12.67ms
[7 ] Iterations:  4000; Final average latency: 12.66ms
[19] Iterations:  4000; Final average latency: 12.70ms
[22] Iterations:  4000; Final average latency: 12.96ms
[17] Iterations:  4000; Final average latency: 13.40ms

real    1m19.263s
user    5m53.360s
sys     1m29.495s
2835 FPS
```

To calculate throughput (frames per second), divide frames by execution time:  
```
224000 / 79 = 2835.44 fps
```

