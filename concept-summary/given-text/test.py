from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama-3_1-8b-q4",
)

print(model.invoke("Who is the president of the united states").content)