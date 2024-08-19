import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime

from langchain_openai import AzureOpenAIEmbeddings  
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path="concept-summary/given-text/.env")

# model = AzureChatOpenAI(
#     azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
#     api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
# )

# ai_msg = model.invoke("""Formuliere den folgenden Absatz in anderen Worten, ohne die Bedeutung zu verändern.:
            

# """)

# print(ai_msg.content)
chunk1 = """Kundenzufriedenheitsbefragungen erfolgen als konstanter Prozess nach 50 % aller Kunden-
anrufe mit der sogenannten After -Call-Befragung, die den Kunden die Möglichkeit gibt, 
Freundlichkeit, fallabschließende Bearbeitung der Anfrage und Gesamteindruck nach dem 
Schulnotenprinzip zu beurteilen .  
 
Festzustellen ist : Das Netz besitzt mehrfach redundante Einspeisungen . 
Zur Unterstützung unserer Erneuerungsstrategie setzen wir ein IT -System zur Bewertung"""

chunk2 = """Kundenzufriedenheitsbefragungen erfolgen als konstanter Prozess nach 50 % aller Kunden-
anrufe mit der sogenannten After -Call-Befragung, die den Kunden die Möglichkeit gibt, 
Freundlichkeit, fallabschließende Bearbeitung der Anfrage und Gesamteindruck nach dem 
Schulnotenprinzip zu beurteilen.   
Turnusmäßig erfolgen ausführliche Telefonbefragungen von Kunden in repräsentativer Grö-
ße durch externe, von der RSN beauftragte Unternehmen."""


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

emb1 = embeddings.embed_query(chunk1)

emb2 = embeddings.embed_query(chunk2)

print(cosine_similarity([emb1, emb2]))