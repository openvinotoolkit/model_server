from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import wikipedia
import time

# generated using wikipedia.random(9)
titles = ['Tennille, Alabama', 'Juan Carlos Lallana', 'Shorrock supercharger', 'Usborne', 'Luc De Vos', 'Plikiai', 'Blak Twang', 'Dog the Bounty Hunter', 'A Reno Divorce']
input = [ Document(page_content=wikipedia.summary(title), metadata={"source": "local"}) for title in titles]

port = 7997
model = "BAAI/bge-small-en-v1.5"
openai_ef = OpenAIEmbeddings(
    base_url=f"http://localhost:{port}/",
    api_key="None",
    model="BAAI/bge-small-en-v1.5",
    tiktoken_enabled=False
            )
chunk_size=1000  # PARAM
chunk_overlap=200  # PARAM
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

texts = text_splitter.split_documents(input)
start = time.time()
db = Chroma.from_documents(texts, openai_ef)

print("Elapsed time: ", time.time() - start, "s")
