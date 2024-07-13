import os
from dotenv import load_dotenv
from create_client import CreateClient

load_dotenv(dotenv_path="concept-summary/rag/.env")

def create_index():
    index_name = os.environ.get("AZURE_AI_SEARCH_INDEX_NAME"),
    index_schema = "./openai-search-demo.json"
    
    start_client = CreateClient(endpoint, key, index_name)
    admin_client = start_client.create_admin_client()
    search_client = start_client.create_search_client()