# Seq2seq demo with python node {#ovms_demo_python_seq2seq}

## Running Model Server on Linux

### Build image

From the root of the repository run:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

### Deploy OpenVINO Model Server with the Python calculator
Prerequisites:
-  image of OVMS with Python support and Optimum installed

Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python node

```bash
cd demos/python_demos/seq2seq_translation
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Running Model Server on Windows (experimental)

### Build Model Server binary with Python enabled
For the Model Server to include embedded Python interpreter you need to [build the binary from scratch](../../../docs/windows_developer_guide.md).

Next steps assume you have followed the building instruction from the developer guide. If you chose different download or installation paths you will need to adjust below instructions.

### Prepare environment for Python servable deployment

Once you have OVMS executable, there are three more steps for the Python servable to work.

1. Create a copy of `pyovms.so` file created during the build with `.pyd` extension in the same location. 

    Go to the location where Python binding building artifacts are stored:

    `cd  C:\git\model_server\bazel-out\x64_windows-opt\bin\src\python\binding`

    Then copy `pyovms.so` to `pyovms.pyd`:

    `cp pyovms.so pyovms.pyd`

2. Set `PYTHONPATH` environment variable for `pyovms` and `openvino` bindings:

    *Windows Command Prompt:*

    `set PYTHONPATH=C:\git\model_server\bazel-out\x64_windows-opt\bin\src\python\binding;C:\opt\intel\openvino\python;%PYTHONPATH%`

    *Windows PowerShell:*

    `$env:PYTHONPATH="C:\git\model_server\bazel-out\x64_windows-opt\bin\src\python\binding;C:\opt\intel\openvino\python;$env:PYTHONPATH"`


3. Switch working directory to python demos and install demo specific Python modules used by the servable:

    ```
    cd C:\git\model_server\demos\python_demos

    python -m pip install -U pip -r requirements.txt
    ```
    **Note**: The building guide required you to download Python 3.9 and if you exactly followed the guide you have it in `C:\opt\Python39`. This is the Python that was used during model server building and OVMS binary links to it - therefore Python servables will use this interpreter. Any `pip` packages used by the servables must be installed to this environment.

### Start the Model Server
With model server binary built and Python environment prepared, go to `seq2seq_translation` demo directory:

`cd seq2seq_translation`

and run model server using executable stored in: `C:\git\model_server\bazel-bin\src\ovms.exe` - you can add `C:\git\model_server\bazel-bin\src` to `Path` environment variable to call `ovms` directly, but assuming it's not added, the launch command would look like:

`C:\git\model_server\bazel-bin\src\ovms.exe --port 9000 --config_path servable/config.json`

To run in the background you can use (via PowerShell):

`Start-Process -NoNewWindow C:\git\model_server\bazel-bin\src\ovms.exe -ArgumentList "--port=9000 --config_path=servable/config.json"`

## Requesting translation
Install client requirements

```bash
pip3 install -r requirements.txt 
```
Run the client script
```bash
python client.py --url localhost:9000
```

Expected output:
```bash
Text:
He never went out without a book under his arm, and he often came back with two.

Translation:
Il n'est jamais sorti sans un livre sous son bras, et il est souvent revenu avec deux.

```
