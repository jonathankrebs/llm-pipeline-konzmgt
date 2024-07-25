import os
from dotenv import load_dotenv

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings  
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_chunks(text, chunk_size=1000, chunk_overlap=200):
   if chunk_size <= chunk_overlap:
      raise ValueError("Chunk size must be greater than overlap size")
   
   chunks = []
   for i in range(0, len(text) - chunk_overlap, chunk_size - chunk_overlap):
      chunks.append(text[i:i + chunk_size])
   return chunks
    
    

load_dotenv(dotenv_path="concept-summary/rag/.env")

azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
vector_store_address: str = f"https://{os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")}.search.windows.net"
vector_store_password: str = os.getenv("AZURE_AI_SEARCH_API_KEY")

# Init embeddings
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

# Init vector store
index_name: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

input_base_path = "concept-summary/rag/input-data/"
input_pdf = "A_full.pdf"

# Parse pdf input
full_path = input_base_path + input_pdf
loader = PyPDFLoader(full_path)
docs = loader.load_and_split()
document_text = ""
for doc in docs:
    document_text += doc.page_content

# Write document text to txt file
tmp_file_path = "concept-summary/rag/output-data/document_text_tmp.txt"
with open(tmp_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(document_text)

# Load txt file in correct format
txt_loader = TextLoader(tmp_file_path, encoding="utf-8")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
txt_docs = txt_loader.load_and_split(text_splitter)


vector_store.add_documents(documents=txt_docs)