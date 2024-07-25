s = "Ich bin ein Mensch"
chunk_size = 4
overlap = 2
if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap size")
    
chunks = []
for i in range(0, len(s) - overlap, chunk_size - overlap):
    chunks.append(s[i:i + chunk_size])

print(chunks)