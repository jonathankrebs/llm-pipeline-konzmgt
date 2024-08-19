from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Init embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
output_base_path = "duplicate-detection/output-data/local/"
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
threshold = 0.84
duplicates = np.argwhere(similarity_matrix > threshold)

output_file_path = output_base_path + "output-base-nomic.txt"
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
    f.write(f"Anzahl der gefundenen Ähnlichkeiten: {count}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Timestamp: {timestamp}\n")