import os
from dotenv import load_dotenv

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings  


load_dotenv(dotenv_path="concept-summary/rag/.env")

azure_endpoint: str = os.getenv("AZURE_OPENAI_ENPOINT")
azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

vector_store_address: str = f"https://{os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")}.search.windows.net"
vector_store_password: str = os.getenv("AZURE_AI_SEARCH_API_KEY")

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=azure_openai_api_key,
)

index_name: str = "langchain-vector-demo"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

# Perform a similarity search
docs = vector_store.similarity_search(
    query="Personalsituation Unternehmen",
    k=3,
    search_type="similarity",
)
print(docs[0].page_content)
# index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME") 
# retriever = AzureAISearchRetriever(
#     content_key="content", top_k=1, index_name=index_name
# )

# retriever.invoke("Personalsituation Unternehmen")