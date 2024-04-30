# Visual RAG using Intel's OVMS

### Step 1. Build docker image

Using default ```README.md```, 

Modify ```Dockerfile.ubuntu``` and add ```RUN pip3 install opencv-python-headless langchain_experimental sentence-transformers open-clip-torch torch``` to install required packages.

Generate openvino modelserver python image. (follow steps mentioned in default ```README.md``` file.

### Step 2. Download models

```bash
cd demos/python_demos/rag_chatbot
export SELECTED_MODEL=llama-2-chat-7b
pip install -r requirements.txt

# download llm model
python download_model.py --model ${SELECTED_MODEL}

# download embedding model 
python download_embedding_model.py --model all-mpnet-base-v
```

### Step 3. start chroma db


```bash
docker run -p 8000:8000 chromadb/chroma
```

### Step 4. Generate Embeddings

```bash
cd rag_chatbot/servable_streams
pip3 install -r requirements.txt 
python3 generate_store_embeddings.py
```

### Step 5. Run model-server and webUI

```bash
# from directory
cd rag_chatbot

docker run -d --rm -p 9000:9000 -v ${PWD}/servable_stream:/workspace -v ${PWD}/${SELECTED_MODEL}:/llm_model \
-v ${PWD}/all-mpnet-base-v2:/embed_model -v ${PWD}/documents:/documents -e SELECTED_MODEL=${SELECTED_MODEL} \
registry.connect.redhat.com/intel/openvino-model-server:py --config_path /workspace/config.json --port 9000

python3 app.py --web_url 10.190.180.102:50055 --ovms_url 10.190.180.102:9000
```