import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime

from langchain_openai import AzureOpenAIEmbeddings  
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path="duplicate-detection/.env")

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

input_base_path = "duplicate-detection/input-data/"
input_pdf = "25_pages.pdf"

# Parse pdf input
full_path = input_base_path + input_pdf
loader = PyPDFLoader(full_path)
docs = loader.load_and_split()
document_text = ""
for doc in docs:
    document_text += doc.page_content

# Write document text to temporary txt file
output_base_path = "duplicate-detection/output-data/openai/"
tmp_file_path = output_base_path + "document_text_tmp.txt"
with open(tmp_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(document_text)

# Load temporary txt file in chunks
txt_loader = TextLoader(tmp_file_path, encoding="utf-8")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = txt_loader.load_and_split(text_splitter)

with open(output_base_path + "chunks.txt", "w", encoding="utf-8") as output_file:
    for chunk in chunks:
        output_file.write(chunk.page_content + "\n")
        output_file.write("-" * 80 + "\n")
   
chunk_embeddings = [embeddings.embed_query(chunk.page_content) for chunk in chunks]

# Identify similar chunks
similarity_matrix = cosine_similarity(chunk_embeddings)
threshold = 0.7
duplicates = np.argwhere(similarity_matrix > threshold)

output_file_path = output_base_path + "output_base.txt"
with open(output_file_path, "w", encoding="utf-8") as f:
    count = 0
    for i, j in duplicates:
        if (i < j & (j - i) > 5):
            count += 1
            f.write(f"Paragraph {i} is similar to Paragraph {j}\n")
            f.write(f"Paragraph {i}: {chunks[i].page_content}\n")
            f.write(f"Paragraph {j}: {chunks[j].page_content}\n")
            f.write(f"Similarity: {similarity_matrix[i, j]}\n")
            f.write("-" * 80 + "\n")
    f.write(f"Anzahl der gefundenen Ähnlichkeiten: {count}\n")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Timestamp: {timestamp}\n")