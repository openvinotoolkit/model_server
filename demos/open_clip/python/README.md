### Prepare model repository

```bash
tree /home/dkalinow/workspace/models/open_clip

/home/dkalinow/workspace/models/open_clip
├── image_encoder_fp32
│   └── 1
│       ├── image_encoder.bin
│       └── image_encoder.xml
└── text_encoder_fp32
    └── 1
        ├── text_encoder.bin
        └── text_encoder.xml

4 directories, 4 files
```

### Prepare config.json

```
{
    "model_config_list": [
        {
            "config": {
                "name": "text_encoder",
                "base_path": "/models/text_encoder_fp32"
            }
        },
        {
            "config": {
                "name": "image_encoder",
                "base_path": "/models/image_encoder_fp32"
            }
        }
    ]
}

```

### Start OVMS

```
docker run -it --rm -p 8913:8913 -p 8914:8914 -v /home/dkalinow/workspace/models/open_clip:/models -v $(pwd)/:/workspace openvino/model_server:2023.0 --port 8913 --rest_port 8914 --config_path /workspace/config.json
```

### Create venv

```
virtualenv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Get cifar10 dataset

```
make
```

### Run client

```
python3 client.py

...
processing dataset/s0000018.webp dataset/s0000018.cls
horse 0.9999999
deer 5.900727e-08
dog 1.7675578e-08
bird 3.783247e-09
cat 1.4346281e-09
automobile 4.835834e-11
ship 2.0260984e-11
truck 5.8079245e-12
frog 3.3908874e-12
airplane 2.9204248e-12
processing dataset/s0000019.webp dataset/s0000019.cls
truck 0.99918765
automobile 0.0005459925
airplane 0.00023471963
deer 2.2064687e-05
bird 5.2667465e-06
frog 2.9750138e-06
dog 5.4120466e-07
horse 4.725381e-07
cat 2.0860007e-07
ship 1.204244e-07
accurracy 95.0 %
```