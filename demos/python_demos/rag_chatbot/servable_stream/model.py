#*****************************************************************************
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

########### Workaround: https://docs.trychroma.com/troubleshooting#sqlite
__import__('pysqlite3')
import sys
import time
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
############
import os
import requests

from pyovms import Tensor

from threading import Thread

from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor
from langchain_community.llms import HuggingFacePipeline
from optimum.intel.openvino import OVModelForCausalLM
import torch
from langchain.chains import RetrievalQA
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed
)

from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS
from ov_embedding_model import OVEmbeddings

FILE = "docs.txt"
TARGET_FOLDER = "/tmp/documents"

SELECTED_MODEL = os.environ.get('SELECTED_MODEL', 'tiny-llama-1b-chat')
LANGUAGE = os.environ.get('LANGUAGE', 'English')
SEED = os.environ.get('SEED')
llm_model_configuration = SUPPORTED_LLM_MODELS[LANGUAGE][SELECTED_MODEL]

EMBEDDING_MODEL = 'all-mpnet-base-v2'
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[EMBEDDING_MODEL]

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Document Splitter
from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, )
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
    "Chinese": ChineseTextSplitter,
}


LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text


text_processor = llm_model_configuration.get(
    "partial_text_processor", default_partial_text_processor
)


def deserialize_prompts(batch_size, input_tensor):
    if batch_size == 1:
        return [bytes(input_tensor).decode()]
    np_arr = deserialize_bytes_tensor(bytes(input_tensor))
    return [arr.decode() for arr in np_arr]


def serialize_completions(batch_size, result):
    if batch_size == 1:
        return [Tensor("completion", result.encode())]
    return [Tensor("completion", serialize_byte_tensor(
        np.array(result, dtype=np.object_)).item())]

def download_documents(file, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    file = open(file, "r")
    for url in file.readlines():
        try:
            url = url.strip()
            if url.find('/'):
                filename = url.rsplit('/', 1)[1]
            else:
                filename = url
            r = requests.get(url, allow_redirects=True, verify=False)
            if r.status_code == 200:
                open(os.path.join(target_folder, filename), 'wb').write(r.content)
                print("Saved", filename, flush=True)
            else:
                print(f"Failed to download: {url}, status code: {r.status_code}", flush=True)
        except Exception as e:
            print(f"Failed to download: {url}, error: {str(e)}", flush=True)

def clean_target_folder(target_folder):
    for file_path in os.listdir(target_folder):
        os.remove(os.path.join(target_folder,file_path))

class OvmsPythonModel:
    def initialize(self, kwargs: dict):

        device = os.environ.get("DEVICE", "CPU")   
        llm_model_dir = os.environ.get("LLM_MODEL_DIR", SELECTED_MODEL)
        llm_model_dir = os.path.join(kwargs["base_path"], llm_model_dir)  # use absolute path
        model_name = llm_model_configuration["model_id"]
        if os.path.isdir(llm_model_dir):
            llm_model_2_load = llm_model_dir
            export = False
        else:
            llm_model_2_load = model_name
            export = True

        print("llm model to load", llm_model_2_load, flush=True)
        self.stop_tokens = llm_model_configuration.get("stop_tokens")
        self.db = None
        class_key = SELECTED_MODEL.split("-")[0]
        self.tok = AutoTokenizer.from_pretrained(llm_model_2_load, trust_remote_code=True)

        if self.stop_tokens is not None:
            if isinstance(self.stop_tokens[0], str):
                self.stop_tokens = self.tok.convert_tokens_to_ids(self.stop_tokens)
            self.stop_tokens = [StopOnTokens(self.stop_tokens)]

        from ov_llm_model import model_classes
        model_class = (
            OVModelForCausalLM
            if not llm_model_configuration["remote"]
            else model_classes[class_key]
        )
        ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
        print(f"Loading LLM model {llm_model_2_load}...class {model_class}", flush=True)
        self.last_refreshDB = 0
        self.base_path = kwargs["base_path"]
        self.active_refresh = True
        self.ov_model = model_class.from_pretrained(
            llm_model_2_load,
            device=device,
            ov_config=ov_config,
            compile=True,
            export=export,
            config=AutoConfig.from_pretrained(llm_model_2_load, trust_remote_code=True),
            trust_remote_code=True)
        print("LLM model loaded", flush=True)

        embedding_model_dir = os.environ.get("EMBEDDING_MODEL_DIR", 'all-mpnet-base-v2')
        embedding_model_dir = os.path.join(kwargs["base_path"], embedding_model_dir)  # use absolute path
        if os.path.isdir(embedding_model_dir):
            embedding_model_2_load = embedding_model_dir
        else:
            embedding_model_2_load = EMBEDDING_MODEL
        print(f"Loading embedding model {embedding_model_2_load}...", flush=True)
        self.embedding = OVEmbeddings.from_model_id(
            embedding_model_2_load,
            do_norm=embedding_model_configuration["do_norm"],
            ov_config={
                "device_name": device,
                "config": {"PERFORMANCE_HINT": "LATENCY"},
            },
            model_kwargs={
                "model_max_length": 512,
            },
        )
        print("Embedding model loaded", flush=True)
        print("Building document database...", flush=True)
        file = os.path.join(self.base_path, FILE)
        if os.path.exists(file):
            self.last_refreshDB = os.stat(file).st_mtime
            download_documents(file, TARGET_FOLDER)
            self.scan_documents(TARGET_FOLDER)

        def refreshDB_if_needed(file, target_folder):
            while self.active_refresh:
                if os.path.exists(file):
                    if (self.last_refreshDB < os.stat(file).st_mtime):
                        print("Refreshing DB", flush=True)
                        clean_target_folder(target_folder)
                        download_documents(file, target_folder)
                        self.scan_documents(target_folder)
                        self.last_refreshDB = os.stat(file).st_mtime
                time.sleep(10)
        
        self.t_refresh = Thread(target=refreshDB_if_needed, args=(file, TARGET_FOLDER,))
        self.t_refresh.start()

        print("Refresh thread started", flush=True)

    def scan_documents(self, target_folder):
        documents = []
        try:
            for file_path in os.listdir(target_folder):
                abs_path = os.path.join(target_folder, file_path)
                print(f"Reading document {abs_path}...", flush=True)
                documents.extend(load_single_document(abs_path))
            print("Documents loaded", flush=True)
            spliter_name = "RecursiveCharacter"  # TODO: Param?
            chunk_size=1000  # TODO: Param?
            chunk_overlap=200  # TODO: Param?
            text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print("Splitting documents...", flush=True)
            self.texts = text_splitter.split_documents(documents)
            print("Documents splitted", self.texts, flush=True)
            if self.db is not None:
                self.db.delete_collection()
            self.db = Chroma.from_documents(self.texts, self.embedding)
            print("Document database built", flush=True)
            vector_search_top_k = 4  # TODO: Param?
            self.retriever = self.db.as_retriever(search_kwargs={"k": vector_search_top_k})
            print("Document database loaded", flush=True)
        except Exception as e:
            print(f"Error occurred while scanning documents: {str(e)}", flush=True)

    def execute(self, inputs: list):
        print("Executing", flush=True)

        batch_size = inputs[0].shape[0]
        if batch_size != 1:
            raise ValueError("Batch size must be 1")
        prompts = deserialize_prompts(batch_size, inputs[0])

        ov_model_exec = self.ov_model.clone()
        streamer = TextIteratorStreamer(
            self.tok, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model=ov_model_exec,
            tokenizer=self.tok,
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
        )
        if self.stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(self.stop_tokens)
          
        pipe = pipeline("text-generation", **generate_kwargs)
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = PromptTemplate.from_template(llm_model_configuration["rag_prompt_template"])
        print("Prompt", prompt, flush=True)
        chain_type_kwargs = {"prompt": prompt}
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs=chain_type_kwargs,
        )

        question = prompts[0]
        def infer(q):
            rag_chain.invoke(q)

        if SEED is not None: set_seed(int(SEED))
        t1 = Thread(target=infer, args=(question,))
        t1.start()

        for new_text in streamer:
            print(new_text, flush=True, end='')
            yield [Tensor("completion", new_text.encode())]

        yield [Tensor("end_signal", "".encode())]
    def finalize(self):
        self.active_refresh = False
        if self.t_refresh:
            self.t_refresh.join()
