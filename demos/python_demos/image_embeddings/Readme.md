# Image Embeddings with OpenVINO Model Server {ovms_image_embeddings}

Image-to-image search system using vision models (CLIP, LAION, DINO) for generating semantic embeddings with OpenVINO Model Server. The client uploads query images and receives similar images from a pre-indexed dataset based on visual content similarity. This enables applications to find visually and semantically related images without requiring text descriptions or manual tagging. The system uses Python code for preprocessing and postprocessing and MediaPipe graphs for optimized inference execution.

## Build image

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

If you want the docker image that supports CPU, run:

```bash
make release_image
```

Else if you want the docker image that supports iGPU's, run:
```bash
make release_image GPU=1
```

# Project Architecture

1. **Model Conversion (`model_conversion/`)**
   - Convert supported multimodal models (e.g., CLIP, Laion, DINO) into **OpenVINO IR format**.
   - Ensures models are optimized for inference on Intel hardware.

2. **Servable Pipeline (`servable/`)**
   - **Preprocessing**: Handles image resizing, normalization(if required).
   - **Postprocessing**: Handles embedding extraction, vector normalization, and formatting.
   - **Config File**: `config_model.json` defines model parameters and pipeline configurations.
   - **MediaPipe Graphs**: Graph definitions for processing inputs/outputs across the 3 models.

3. **gRPC CLI (`grpc_cli.py`)**
   - Iterates over a folder of images.
   - Extracts embeddings using the OpenVINO-served models.
   - Stores embeddings in a **Vector Database** (Qdrant)

4. **Search API (`search_images.py`)**
   - Accepts an input query image.
   - Generates its embedding and queries the Vector DB.
   - Returns the most similar images based on cosine similarity or other distance metrics.

5. **Search App (`streamlit_app.py`)**
   - Provided a frontend for the users to interact with the project and test it
   - Allows users to upload images and perform semantic search


## Installation and setup

```bash
cd demos/python_demos/image_embeddings
python3 -m venv venv
pip install -r requirements.txt
```


## Model conversion
```bash
python3 -m venv venv
cd demos/python_demos/image_embeddings/model_conversion
```

For clip
```bash
python clip_conversion.py
```

For laion
```bash
python laion_conversion.py
```

For Dino
```bash
python dino_conversion.py
```

## Deploying OpenVINO Model Server
Prerequisites:
-  image of OVMS with Python support and Optimum installed
Mount the `./servable` which contains:
- `post.py` and `pre.py` - python scripts which are required for execution.
- `config_model.json` - which defines which servables should be loaded.
- `graph_clip.pbtxt`, `graph_laion.pbtxt`, `graph_dino.pbtxt` - which defines MediaPipe graph containing python nodes.


```bash
cd demos/python_demos/image_embeddings
```


To use CPU
``bash
docker run -it --rm \
-p 9000:9000 -p 8000:8000 \
-v ${PWD}/servable:/workspace \
-v ${PWD}/model_conversion/saved_mod/siglip:/saved_mod/dino \
-v ${PWD}/model_conversion/saved_mod/clip:/saved_mod/clip \
-v ${PWD}/model_conversion/saved_mod/laion:/saved_mod/laion \
openvino/model_server:py \
--config_path /workspace/config_model.json \
--port 9000 --rest_port 8000

```

To use GPU
```bash
docker run -it --rm \
  --device=/dev/dxg \
  --volume /usr/lib/wsl:/usr/lib/wsl \
  -p 9000:9000 -p 8000:8000 \
  -v ${PWD}/servable:/workspace \
  -v ${PWD}/model_conversion/saved_mod/dino:/saved_mod/dino \
  -v ${PWD}/model_conversion/saved_mod/clip:/saved_mod/clip \
  -v ${PWD}/model_conversion/saved_mod/laion:/saved_mod/laion \
  ovms-gpu-custom \
    --config_path /workspace/config_model.json \
    --port 9000 \
    --rest_port 8000
```

## Deploying the Vector Database

The next step is to start the vector database. In this project, we are using **Qdrant**, an open-source vector database optimized for efficient semantic search.

Run the following command to start Qdrant with Docker:

```bash
cd demos/python_demos/image_embeddings
docker run -p 6333:6333 qdrant/qdrant
```

## Running the Demo

Once the vector database is running and the OpenVINO Model Server is deployed, open another terminal and run:

```bash
source venv/bin/activate
cd demos/python_demos/image_embeddings
python grpc_cli.py
or
python grpc_cli.py --model "your selected model name"
```
Once you run this you should see something like this
```bash
Building Image Database
==================================================
Waiting for server to be ready...
Server Ready Check:   0%|                                                                                                                             | 0/15 [00:00<?, ?sec/s]Server is ready!
Server Ready Check: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 207.60sec/s]

Select a model to use:
1. clip_graph
2. dino_graph
3. laion_graph
Enter the number corresponding to the model: 1
Saved model selection: clip_graph

✓ Found 106 images in './demo_images'

Processing images and generating embeddings...
Processing Images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 106/106 [00:05<00:00, 20.61img/s]

Uploading embeddings to Qdrant...
Uploading to Database: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 16.28batch/s]

Database Build Complete!
Statistics:
   • Images processed: 106
   • Average inference time: 43.66 ms
   • Total processing time: 5.21 s
   • Throughput: 20.33 images/sec
   • Collection: 'image_embeddings'

Database built successfully!
Total images processed: 106
Model saved for future searches: clip_graph
```

In this step, the images stored in the folder are converted into embeddings and saved in Qdrant. Once this is done, we can perform image-to-image search for any image placed in the directory against the ones stored in the vector database.

Next up run
```bash
python search_images.py ./demo_img.jpg 5
python search_images.py ./dem2.jpg 3
```
You should see
```bash
Searching for Similar Images
==================================================
Query: ./demo_img.jpg
Auto-loaded Model: clip_graph
Top K: 5

Top 5 similar images:
--------------------------------------------------
1. mountains.jpg (similarity: 0.744)
2. img_87.jpg (similarity: 0.648)
3. img_23.jpg (similarity: 0.638)
4. img_48.jpg (similarity: 0.630)
5. oldesttree.jpg (similarity: 0.616)

Results saved to: ./similar_images

Performance Summary
--------------------------------------------------
Embedding Inference Latency : 155.94 ms
Qdrant Search Latency       : 32.01 ms
End-to-End Latency          : 2165.65 ms
```
From this, you can see that the top 5 most relevant images are stored in the similar_images folder in the repository. You can also change the value of k to adjust how many top results are returned, depending on your usage.

If you want a frontend interface that allows you to upload images, run the following command:

```bash
streamlit run streamlit_app.py
```