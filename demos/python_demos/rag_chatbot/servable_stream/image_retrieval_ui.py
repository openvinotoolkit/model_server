import streamlit as st
from vector_stores import db
from utils import config_reader as reader

if 'config' not in st.session_state.keys():
    st.session_state.config = reader.read_config('config.yaml')

st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

st.title("Image Retriever")

@st.cache_resource       
def load_db():
    if 'vs' not in st.session_state.keys():
        with st.spinner('Loading vector-store chain'):
            host = st.session_state.config['vector_db']['host']
            port = int(st.session_state.config['vector_db']['port'])
            selected_db = st.session_state.config['vector_db']['choice_of_db']
            st.session_state['vs'] = db.VS(host, port, selected_db)
            
load_db()
            
def custom_image_retrieval(Q, N=4):
    print (Q)
    
    image_config = {"configurable": {"k_image_docs": {"k": N}}}
    results = st.session_state.vs.image_retriever.invoke(Q, config=image_config)
    
    print (len(results))
    
    col1, col2 = st.columns(2)

    image_paths = []
    captions = []
    
    for r in results:
        image_paths.append(r.metadata['frame_path'])
        captions.append(r.metadata['video'])

    for i, image in enumerate(image_paths):
        if i % 2 == 0:
            with col1:
                st.image(image, caption=captions[i])
        else:
            with col2:
                st.image(image, caption=captions[i])

N = st.slider('Number of Images', 1, 16, 2)
N = int(N)
Q = st.text_input('Enter Text to retrieve Images')

if Q is not None and Q != '':
    custom_image_retrieval(Q, N)