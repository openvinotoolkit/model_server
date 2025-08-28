import sys
sys.path.append("../../common/python")
import tritonclient.grpc as grpcclient

import numpy as np
from pathlib import Path
import os
import shutil
import json
import time
from qdrant_client import QdrantClient

# ------------------------ Settings ------------------------
IMAGE_FOLDER = "./demo_images"
QDRANT_COLLECTION = "image_embeddings"
OUTPUT_DIR = "./similar_images"
MODEL_CONFIG_FILE = "./selected_model.json"

def load_selected_model():
    """Load the previously selected model"""
    if os.path.exists(MODEL_CONFIG_FILE):
        with open(MODEL_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get("selected_model")
    return None

def get_query_embedding(query_image_path, model_name):
    """Get embedding for query image via gRPC"""
    client = grpcclient.InferenceServerClient("localhost:9000")

    with open(query_image_path, "rb") as f:
        image_data = f.read()

    image_np = np.array([image_data], dtype=np.object_)
    image_input = grpcclient.InferInput("image", [1], "BYTES")
    image_input.set_data_from_numpy(image_np)

    start = time.perf_counter()
    results = client.infer(model_name, [image_input])
    end = time.perf_counter()

    embedding = results.as_numpy('embedding')[0]
    latency_ms = (end - start) * 1000  # milliseconds

    return embedding, latency_ms

def search_similar_images(query_embedding, top_k=5):
    """Search for similar images in Qdrant"""
    qdrant = QdrantClient("localhost", port=6333)

    start = time.perf_counter()
    search_result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding.tolist(),
        limit=top_k
    )
    end = time.perf_counter()

    search_latency_ms = (end - start) * 1000
    return search_result.points, search_latency_ms

def save_results(search_results, output_dir):
    """Save similar images to output directory"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    print(f"\nTop {len(search_results)} similar images:")
    print("-" * 50)

    for i, result in enumerate(search_results):
        filename = result.payload["filename"]
        similarity = result.score

        src_path = os.path.join(IMAGE_FOLDER, filename)
        dst_path = os.path.join(output_dir, f"rank_{i+1}_score_{similarity:.3f}_{filename}")

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"{i+1}. {filename} (similarity: {similarity:.3f})")
        else:
            print(f"{i+1}. {filename} (similarity: {similarity:.3f}) File not found")

    print(f"\nResults saved to: {output_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python search_images.py <query_image_path> [top_k]")
        print("Example: python search_images.py ./mt.jpg 5")
        return

    query_image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not os.path.exists(query_image_path):
        print(f"Query image not found: {query_image_path}")
        return

    # Auto-load model
    model_name = load_selected_model()
    if not model_name:
        print("No model selection found. Please run grpc_cli.py first!")
        return

    try:
        print("Searching for Similar Images")
        print("="*50)
        print(f"Query: {query_image_path}")
        print(f"Auto-loaded Model: {model_name}")
        print(f"Top K: {top_k}")

        total_start = time.perf_counter()

        # Get query embedding + latency
        query_embedding, infer_latency = get_query_embedding(query_image_path, model_name)

        # Search similar images + latency
        results, search_latency = search_similar_images(query_embedding, top_k)

        total_end = time.perf_counter()
        total_latency = (total_end - total_start) * 1000

        # Save results
        save_results(results, OUTPUT_DIR)

        print("\nPerformance Summary")
        print("-" * 50)
        print(f"Embedding Inference Latency : {infer_latency:.2f} ms")
        print(f"Qdrant Search Latency       : {search_latency:.2f} ms")
        print(f"End-to-End Latency          : {total_latency:.2f} ms")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
