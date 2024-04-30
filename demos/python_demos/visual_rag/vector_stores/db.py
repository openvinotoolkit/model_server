import chromadb
from langchain_community.vectorstores import VDMS
from langchain_community.vectorstores.vdms import VDMS_Client
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from typing import List, Optional, Iterable
from langchain_core.runnables import ConfigurableField

class VS:
    
    def __init__(self, host, port, selected_db):
        self.host = host
        self.port = port
        self.selected_db = selected_db
        
        # initializing important variables
        self.client = None
        self.text_db = None
        self.image_db = None
        self.text_embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
        self.image_embedder = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
        self.text_collection = 'text-test'
        self.image_collection = 'image-test'
        self.text_retriever = None
        self.image_retriever = None
        
        # initialize_db
        self.get_db_client()
        self.init_db()
        
    def get_db_client(self):
        
        if self.selected_db == 'chroma':
            print ('Connecting to Chroma db server . . .')
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
        
        if self.selected_db == 'vdms':
            print ('Connecting to VDMS db server . . .')
            self.client = VDMS_Client(host=self.host, port=self.port)

    def init_db(self):
        print ('Loading db instances')
        if self.selected_db ==  'chroma':
            self.text_db = Chroma(
                client = self.client,
                embedding_function = self.text_embedder,
                collection_name = self.text_collection,
            )

            self.image_db = Chroma(
                client = self.client,
                embedding_function = self.image_embedder,
                collection_name = self.image_collection,
            )

        if self.selected_db == 'vdms':
            self.text_db = VDMS (
                client = self.client,
                embedding = self.text_embedder,
                collection_name = self.text_collection,
                engine = "FaissFlat",
            )

            self.image_db = VDMS (
                client = self.client,
                embedding = self.image_embedder,
                collection_name = self.image_collection,
                engine = "FaissFlat",
            )

        self.image_retriever = self.image_db.as_retriever(search_type="mmr").configurable_fields(
            search_kwargs=ConfigurableField(
                id="k_image_docs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )
        
        self.text_retriever = self.text_db.as_retriever().configurable_fields(
            search_kwargs=ConfigurableField(
                id="k_text_docs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )
        
    def length(self):
        if self.selected_db == 'chroma':
            texts = self.text_db.__len__()
            images = self.image_db.__len__()
            return (texts, images)
        
        if self.selected_db == 'vdms':
            pass
        
        return (None, None)
        
    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name=collection_name)
        
    def add_images(
            self,
            uris: List[str],
            metadatas: Optional[List[dict]] = None,
        ):

        self.image_db.add_images(uris, metadatas)

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
        ):

        self.text_db.add_texts(texts, metadatas)

    def MultiModalRetrieval(
            self,
            query: str,
            n_texts: Optional[int] = 1,
            n_images: Optional[int] = 3,
        ):
        
        text_config = {"configurable": {"k_text_docs": {"k": n_texts}}}
        image_config = {"configurable": {"k_image_docs": {"k": n_images}}}

        text_results = self.text_retriever.invoke(query, config=text_config)
        image_results = self.image_retriever.invoke(query, config=image_config)

        return text_results + image_results