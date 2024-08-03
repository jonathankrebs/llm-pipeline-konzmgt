import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

input_base_path = "concept-summary/rag/input-data/datteln/eon/"
input_pdf = "NBK_EON_Koop.pdf"

# Parse pdf input
full_path = input_base_path + input_pdf
loader = PyPDFLoader(full_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
all_splits = loader.load_and_split(text_splitter)

# Create embeddings
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=local_embeddings)

query = "Personalsituation Unternehmen"
docs = vectorstore.similarity_search(
    query=query,
    k=3
)


print(docs)