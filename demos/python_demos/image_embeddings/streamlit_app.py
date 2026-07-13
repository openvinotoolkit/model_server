import os
import io
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# Triton/OVMS (gRPC)
import tritonclient.grpc as grpcclient
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# ------------------------ Config ------------------------
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", "./demo_images")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
TRITON_URL = os.environ.get("TRITON_URL", "localhost:9000")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "image_embeddings")
MODEL_CONFIG_FILE = os.environ.get("MODEL_CONFIG_FILE", "./selected_model.json")
AVAILABLE_MODELS = ["clip_graph", "dino_graph", "laion_graph"]

# Model descriptions for better UX
MODEL_DESCRIPTIONS = {
    "clip_graph": "CLIP - Great for general image-text understanding",
    "dino_graph": "DINO - Excellent for object detection and features",
    "laion_graph": "LAION - Trained on large-scale web data"
}

# ------------------------ Custom CSS ------------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* More specific main app styling */
    .main .block-container {
        padding: 1rem 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    /* Search section - Remove white background */
    .search-section {
        background: transparent;
        padding: 1.5rem;
        border-radius: 10px;
        border: none;
        margin-bottom: 1rem;
    }

    /* Results grid - More specific selectors */
    div[data-testid="column"] .result-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }

    div[data-testid="column"] .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* Stats and metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    /* Status indicators */
    .status-success { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Custom buttons - More specific */
    div[data-testid="column"] .stButton > button {
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    div[data-testid="column"] .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Fix for white blocks - ensure all containers are transparent */
    div[data-testid="stImage"] {
        background: transparent !important;
    }

    /* Ensure columns display properly */
    div[data-testid="column"] {
        background: transparent !important;
    }

    /* Fix for selectbox and other widgets */
    .stSelectbox, .stFileUploader, .stSlider {
        background: transparent !important;
    }

    /* Ensure expanders work correctly */
    .streamlit-expanderHeader {
        background: transparent !important;
    }

    /* Remove any white backgrounds from containers */
    .element-container {
        background: transparent !important;
    }

    /* Fix file uploader area */
    .stFileUploader > div {
        background: transparent !important;
    }

    /* Remove white backgrounds from form elements */
    .stMarkdown, .stText {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------ Helpers ------------------------
@st.cache_resource(show_spinner=False)
def get_triton_client():
    return grpcclient.InferenceServerClient(TRITON_URL)

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

def check_system_status() -> Tuple[bool, bool]:
    """Check if Triton and Qdrant are accessible"""
    triton_ok = False
    qdrant_ok = False

    try:
        client = get_triton_client()
        client.is_server_ready()
        triton_ok = True
    except Exception:
        pass

    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        qdrant_ok = any(c.name == QDRANT_COLLECTION for c in collections.collections)
    except Exception:
        pass

    return triton_ok, qdrant_ok

def load_selected_model_from_file() -> Optional[str]:
    if os.path.exists(MODEL_CONFIG_FILE):
        try:
            with open(MODEL_CONFIG_FILE, "r") as f:
                return json.load(f).get("selected_model")
        except Exception:
            return None
    return None

def save_selected_model_to_file(model_name: str):
    with open(MODEL_CONFIG_FILE, "w") as f:
        json.dump({"selected_model": model_name}, f)

def embed_image_bytes(img_bytes: bytes, model_name: str) -> np.ndarray:
    """Send image bytes to OVMS via gRPC and get embedding."""
    client = get_triton_client()
    infer_input = grpcclient.InferInput("image", [1], "BYTES")
    infer_input.set_data_from_numpy(np.array([img_bytes], dtype=np.object_))
    result = client.infer(model_name, [infer_input])
    embedding = result.as_numpy("embedding")[0]
    return embedding

def qdrant_search(embedding: np.ndarray, top_k: int = 6):
    qdrant = get_qdrant_client()
    try:
        # Try new API first
        out = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding.tolist(),
            limit=top_k,
        )
        return out.points
    except Exception:
        # Fallback to old API
        out = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=embedding.tolist(),
            limit=top_k,
        )
        return out

def list_dataset_images(folder: str) -> List[Path]:
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]

def get_collection_stats():
    """Get statistics about the Qdrant collection"""
    try:
        client = get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            'total_vectors': info.vectors_count or 0,
            'status': info.status or 'unknown'
        }
    except Exception:
        return None

# ------------------------ UI Components ------------------------
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🔎 Image Similarity Search</h1>
        <p>Powered by OVMS + Qdrant • Find visually similar images using AI embeddings</p>
    </div>
    """, unsafe_allow_html=True)

def render_system_status():
    """Display system status in sidebar"""
    triton_ok, qdrant_ok = check_system_status()

    st.markdown("### 🔧 System Status")

    col1, col2 = st.columns(2)
    with col1:
        status_class = "status-success" if triton_ok else "status-error"
        status_text = "Online" if triton_ok else "Offline"
        st.markdown(f'**Triton/OVMS**<br/><span class="{status_class}">● {status_text}</span>',
                   unsafe_allow_html=True)

    with col2:
        status_class = "status-success" if qdrant_ok else "status-error"
        status_text = "Ready" if qdrant_ok else "Not Ready"
        st.markdown(f'**Qdrant DB**<br/><span class="{status_class}">● {status_text}</span>',
                   unsafe_allow_html=True)

    # Collection stats
    if qdrant_ok:
        stats = get_collection_stats()
        if stats:
            vector_count = stats['total_vectors'] or 0
            status = stats['status'] or 'unknown'
            st.markdown("### 📊 Collection Stats")
            st.markdown(f"""
            <div class="metric-card">
                <strong>{vector_count:,}</strong> vectors indexed<br/>
                Status: <span class="status-success">{status}</span>
            </div>
            """, unsafe_allow_html=True)

def render_model_selector():
    """Enhanced model selection with descriptions"""
    st.markdown("### 🤖 AI Model")

    # Try to auto-load previous selection
    default_model = load_selected_model_from_file()
    if default_model and default_model in AVAILABLE_MODELS:
        default_idx = AVAILABLE_MODELS.index(default_model)
    else:
        default_idx = 0

    # Create options with descriptions
    options = [f"{model} - {MODEL_DESCRIPTIONS.get(model, 'AI embedding model')}"
              for model in AVAILABLE_MODELS]

    selected_option = st.selectbox(
        "Choose embedding model:",
        options,
        index=default_idx,
        help="Different models excel at different types of image understanding"
    )

    # Extract model name from selection
    model_name = selected_option.split(" - ")[0]

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save Default", help="Remember this model choice"):
            save_selected_model_to_file(model_name)
            st.success(f"✅ Saved: {model_name}")

    return model_name

def render_search_interface():
    """Enhanced search interface - without white background"""
    # Remove the div wrapper that was causing the white block
    st.markdown("### 📤 Upload Query Image")

    # File uploader with better styling
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload an image to find visually similar ones"
    )

    st.markdown("**OR**")

    # Dataset image selector
    st.markdown("### 📂 Choose from Dataset")
    dataset_images = list_dataset_images(IMAGE_FOLDER)

    if dataset_images:
        # Show thumbnail preview grid for dataset selection
        st.write(f"Found {len(dataset_images)} images in dataset:")

        # Create a selectbox with image names
        pick = st.selectbox(
            "Select dataset image:",
            ["— Select an image —"] + [p.name for p in dataset_images],
            help="Pick from pre-indexed images in your dataset"
        )

        # Show preview of selected dataset image
        if pick != "— Select an image —":
            selected_path = Path(IMAGE_FOLDER) / pick
            if selected_path.exists():
                with st.expander("🔍 Preview Selected Image", expanded=True):
                    img = Image.open(selected_path)
                    st.image(img, caption=pick, use_container_width=True)
    else:
        st.warning(f"📁 No images found in `{IMAGE_FOLDER}`")
        pick = "— Select an image —"

    # Determine query image
    query_img_bytes = None
    query_img_label = None

    if uploaded is not None:
        query_img_bytes = uploaded.read()
        query_img_label = uploaded.name
    elif pick != "— Select an image —":
        fp = Path(IMAGE_FOLDER) / pick
        if fp.exists():
            query_img_bytes = fp.read_bytes()
            query_img_label = fp.name

    return query_img_bytes, query_img_label

def render_search_results(results, search_time: float):
    """Fixed results display with proper image loading"""
    st.markdown("### 🎯 Search Results")

    if not results:
        st.warning("🤷 No similar images found. Make sure the Qdrant collection is populated!")
        return

    # Results header with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results Found", len(results))
    with col2:
        st.metric("Search Time", f"{search_time:.2f}s")
    with col3:
        try:
            best_score = max([r.score if hasattr(r, 'score') and r.score is not None else 0 for r in results])
        except:
            best_score = 0
        st.metric("Best Match", f"{best_score:.3f}")

    st.markdown("---")

    # Results grid with fixed image loading
    num_cols = min(3, len(results))
    cols = st.columns(num_cols)

    for i, result in enumerate(results):
        try:
            # Handle different Qdrant response formats
            if hasattr(result, 'payload') and result.payload:
                filename = result.payload.get("filename", "unknown")
            elif hasattr(result, 'metadata') and result.metadata:
                filename = result.metadata.get("filename", "unknown")
            else:
                filename = "unknown"

            # Get score safely
            score = getattr(result, "score", 0) if hasattr(result, "score") else 0

            # Build image path
            img_path = Path(IMAGE_FOLDER) / filename

            with cols[i % num_cols]:
                st.markdown(f'''
                <div class="result-card">
                    <h4>#{i+1} Match</h4>
                    <p><strong>Score:</strong> {score:.4f}</p>
                    <p><strong>File:</strong> {filename}</p>
                </div>
                ''', unsafe_allow_html=True)

                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True, caption=f"Score: {score:.4f}")

                        # Additional metadata if available
                        with st.expander("📄 Details"):
                            st.write(f"**Filename:** `{filename}`")
                            st.write(f"**Similarity Score:** `{score:.6f}`")
                            st.write(f"**File Size:** `{img_path.stat().st_size / 1024:.1f} KB`")
                            st.write(f"**Dimensions:** `{img.size[0]}x{img.size[1]}`")
                    except Exception as img_error:
                        st.error(f"❌ Error loading image: {img_error}")
                else:
                    st.error(f"❌ Image file not found: {filename}")

        except Exception as e:
            st.error(f"Error processing result {i}: {e}")

# ------------------------ Main App ------------------------
def main():
    st.set_page_config(
        page_title="Image Similarity Search",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_custom_css()

    # Header
    render_header()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        render_system_status()
        st.markdown("---")
        model_name = render_model_selector()

        st.markdown("---")
        st.markdown("### 🎚️ Search Settings")
        top_k = st.slider(
            "Number of results:",
            min_value=1,
            max_value=20,
            value=6,
            help="How many similar images to return"
        )

        st.markdown("---")
        st.markdown("### 📡 Connection Info")
        st.code(f"""
Host: {QDRANT_HOST}:{QDRANT_PORT}
Triton: {TRITON_URL}
Collection: {QDRANT_COLLECTION}
        """)

    # Main content
    col_search, col_results = st.columns([1, 2])

    with col_search:
        query_img_bytes, query_img_label = render_search_interface()

        # Show query image preview
        if query_img_bytes:
            st.markdown("### 🖼️ Query Image")
            query_img = Image.open(io.BytesIO(query_img_bytes))
            st.image(query_img, caption=f"Query: {query_img_label}", use_container_width=True)

            # Search button
            search_button = st.button(
                "🔍 Find Similar Images",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("👆 Upload an image or select one from the dataset to begin searching")
            search_button = False

    with col_results:
        if search_button and query_img_bytes:
            try:
                with st.spinner("🔄 Computing embeddings and searching..."):
                    start_time = time.time()

                    # Progress bar for better UX
                    progress = st.progress(0)
                    progress.progress(25, "Computing image embedding...")

                    emb = embed_image_bytes(query_img_bytes, model_name)
                    progress.progress(75, "Searching similar images...")

                    results = qdrant_search(emb, top_k=top_k)
                    progress.progress(100, "Complete!")

                    search_time = time.time() - start_time
                    progress.empty()

                render_search_results(results, search_time)

            except Exception as e:
                st.error("❌ Search failed!")
                st.exception(e)

                # Helpful troubleshooting
                st.markdown("### 🔧 Troubleshooting")
                triton_ok, qdrant_ok = check_system_status()
                st.write(f"**Triton/OVMS:** {'✅ OK' if triton_ok else '❌ Failed'}")
                st.write(f"**Qdrant:** {'✅ OK' if qdrant_ok else '❌ Failed'}")

                st.markdown("""
                **Common Issues:**
                - Check if Triton/OVMS server is running
                - Verify Qdrant database is accessible
                - Ensure the collection exists and has data
                - Confirm the model name matches your deployment
                - Check IMAGE_FOLDER path contains the indexed images
                """)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <h3>🎯 Ready to Search</h3>
                <p>Upload an image or select from dataset to find visually similar images</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()