# MediaPipe Multi Model Demo {#ovms_docs_demo_mediapipe_multi_model}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.
It is using a sequence of models:
```protobuf
input_stream: "in1"
input_stream: "in2"
output_stream: "out"
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:dummy"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "dummy"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:add"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "add"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:dummy"
  input_stream: "DUMMY_IN:in1"
  output_stream: "DUMMY_OUT:dummy_output"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
        tag_to_input_tensor_names {
          key: "DUMMY_IN"
          value: "b"
        }
        tag_to_output_tensor_names {
          key: "DUMMY_OUT"
          value: "a"
        }
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:add"
  input_stream: "ADD_INPUT1:dummy_output"
  input_stream: "ADD_INPUT2:in2"
  output_stream: "SUM:out"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
        tag_to_input_tensor_names {
          key: "ADD_INPUT1"
          value: "input1"
        }
        tag_to_input_tensor_names {
          key: "ADD_INPUT2"
          value: "input2"
        }
        tag_to_output_tensor_names {
          key: "SUM"
          value: "sum"
        }
    }
  }
}
```

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip 

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

## Prepare models

Clone the repository and enter mediapipe image_classification directory
```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/multi_model_graph
```

## Download models
### On Linux
```bash
cp -r ../../../src/test/add_two_inputs_model ./dummyAdd/
cp -r ../../../src/test/dummy ./dummyAdd/
```

### On Windows
```bat
xcopy /s /e /q /y ..\..\..\src\test\add_two_inputs_model .\dummyAdd\add_two_inputs_model\
xcopy /s /e /q /y ..\..\..\src\test\dummy .\dummyAdd\dummy\
```
 

## Server Deployment
:::{dropdown} **Deploying with Docker**
Prepare virtualenv according to [kserve samples readme](https://github.com/openvinotoolkit/model_server/blob/releases/2025/2/client/python/kserve-api/samples/README.md)
```bash
docker run -d -v $PWD:/mediapipe -p 9000:9000 openvino/model_server:latest --config_path /mediapipe/config.json --port 9000
```
:::
:::{dropdown} **Deploying on Bare Metal**
Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.
```bat
ovms --config_path config.json --port 9000
```
:::
## Run the client:
```console
python mediapipe_multi_model_client.py --grpc_port 9000
Output:
[[ 3.  5.  7.  9. 11. 13. 15. 17. 19. 21.]]
```
