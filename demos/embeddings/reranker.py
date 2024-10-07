# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "state_of_the_union.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OpenVINOEmbeddings(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2"
)
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

#query = "What did the president say about Ketanji Brown Jackson"
#docs = retriever.invoke(query)
#pretty_print_docs(docs)

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker

model_name = "BAAI/bge-reranker-large"

ov_compressor = OpenVINOReranker(model_name_or_path=model_name, top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=ov_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

#https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_compressors/openvino_rerank.py#L132
#https://github.com/langchain-ai/langchain/blob/907c758d67764385828c8abad14a3e64cf44d05b/docs/docs/integrations/document_transformers/openvino_rerank.ipynb


"""
        print(self.tokenizer)
        print(len(query_passage_pairs))
        for k,v in input_tensors.items():
            print(k, v.shape)


XLMRobertaTokenizerFast(name_or_path='BAAI/bge-reranker-large', vocab_size=250002, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
        0: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        250001: AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True, special=True),
}
20
input_ids torch.Size([20, 134])
attention_mask torch.Size([20, 134])
"""
