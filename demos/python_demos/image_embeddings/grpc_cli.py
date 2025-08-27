import sys
sys.path.append("../../common/python")
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

import numpy as np
import os
import grpc
import time
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# ------------------------ Settings ------------------------
IMAGE_FOLDER = "./demo_images"
QDRANT_COLLECTION = "image_embeddings"
MODEL_CONFIG_FILE = "./selected_model.json"  # New: Save model choice


def setup_grpc_client():
    """Setup GRPC client and wait for server ready"""
    url = "localhost:9000"
    client = grpcclient.InferenceServerClient(url)
    channel = grpc.insecure_channel(url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Wait for server ready
    timeout = 15
    while timeout:
        request = service_pb2.ServerReadyRequest()
        response = grpc_stub.ServerReady(request)
        print("Server Ready: {}".format(response.ready))
        if response.ready:
            break
        time.sleep(1)
        timeout -= 1

    if not response.ready:
        print("Models are not ready. Increase timeout or check server setup.")
        exit(-1)

    return client


def select_model():
    """Interactive model selection"""
    available_models = ["clip_graph", "dino_graph", "laion_graph"]
    print("\nSelect a model to use:")
    for i, model in enumerate(available_models):
        print(f"{i + 1}. {model}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to the model: "))
            if 1 <= choice <= len(available_models):
                selected_model = available_models[choice - 1]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")

    print(f"\nYou selected: {selected_model}")

    # Save model selection
    with open(MODEL_CONFIG_FILE, 'w') as f:
        json.dump({"selected_model": selected_model}, f)
    print(f"Saved model selection: {selected_model}")

    return selected_model


def build_database(client, selected_model):
    """Build/Update the image database"""
    qdrant = QdrantClient("localhost", port=6333)
    vector_size = None
    inference_times = []  # Track per-image inference times
    total_start = time.perf_counter()  # For total time

    # Create collection if not exists
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        print(f"Creating Qdrant collection: {QDRANT_COLLECTION}")

    # Process all images in demo folder
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nFound {len(image_files)} images in '{IMAGE_FOLDER}'")

    points = []

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        with open(img_path, "rb") as f:
            image_data = f.read()

        image_np = np.array([image_data], dtype=np.object_)
        image_input = grpcclient.InferInput("image", [1], "BYTES")
        image_input.set_data_from_numpy(image_np)

        # --- measure inference time ---
        start_time = time.perf_counter()
        results = client.infer(selected_model, [image_input])
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        inference_times.append(elapsed_ms)
        # --------------------------------

        embedding = results.as_numpy('embedding')[0]

        if vector_size is None:
            vector_size = embedding.shape[0]
            if not qdrant.collection_exists(QDRANT_COLLECTION):
                qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
                )

        points.append(
            rest.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"filename": img_name}
            )
        )

    # Upload to Qdrant
    if points:
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

    total_end = time.perf_counter()
    total_time = total_end - total_start
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    throughput = len(inference_times) / total_time if total_time > 0 else 0

    print(f"\nInserted {len(points)} embeddings into Qdrant collection '{QDRANT_COLLECTION}'")
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f"Total Processing Time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} images/sec")

    return len(points)


def main():
    print("Building Image Database")
    print("=" * 50)

    try:
        # Setup
        client = setup_grpc_client()
        selected_model = select_model()

        # Build database
        count = build_database(client, selected_model)

        print(f"\nDatabase built successfully!")
        print(f"Total images processed: {count}")
        print(f"Model saved for future searches: {selected_model}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
