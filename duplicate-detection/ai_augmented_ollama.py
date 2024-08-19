from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
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

model_name = "llama-3_1-8b-q8"
model = ChatOllama(
    model=model_name,
)

ai_chunks = []
# Replace chunks with ai-summary
with open(output_base_path + "chunks.txt", "w", encoding="utf-8") as output_file:
    for chunk in chunks:
        # Set the system message and prompt
        messages = [
            (
                "system",
                """You are a helpful assistant that summarizes a given text. Make sure to only include the most relevant arguments of the given text. Use bullet points for the summary. Only Output the summary and nothing else. Don't write 'Here is your summary' or 'Hier ist eine Zusammenfassung' or anything alike. Your answer starts and ends with the relevant bullet points, e.g. 
                    '- Information 1
                    - Information 2'
                """,
            ),
            ("human", chunk.page_content),
        ]
        # Generate output
        ai_msg = model.invoke(messages)
        ai_chunks.append(ai_msg.content)
        output_file.write(ai_msg.content + "\n")
        output_file.write("-" * 80 + "\n")
   
chunk_embeddings = [embeddings.embed_query(chunk) for chunk in ai_chunks]

# Identify similar chunks
similarity_matrix = cosine_similarity(chunk_embeddings)
threshold = 0.84
duplicates = np.argwhere(similarity_matrix > threshold)

output_file_path = output_base_path + "output-ai-augmented-llama-3_1-8B-q8.txt"
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