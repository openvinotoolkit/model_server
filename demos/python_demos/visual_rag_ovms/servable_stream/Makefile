install:
	pip1 instll -r requirements.txt

generate-image-embeddings:	
	python3 generate_image_embeddings.py

generate-text-embeddings:
	python3 generate_text_embeddings.py

run-image-retrieval-ui:
	streamlit run image_retrieval_ui.py --server.address 0.0.0.0 --server.port 50055

run-visual-rag:
	streamlit run visual-rag-ui.py --server.address 0.0.0.0 --server.port 50055

run:
	docker run -d -p 8000:8000 chromadb/chroma
	pip3 install -r requirements.txt
	python3 generate_store_embeddings.py
	streamlit run image_retrieval_ui.py --server.address 0.0.0.0 --server.port 50022 &
	streamlit run visual-rag-ui.py --server.address 0.0.0.0 --server.port 50055 &
	echo 'Started image ui at port 50022 and visual rag ui at port 50055'