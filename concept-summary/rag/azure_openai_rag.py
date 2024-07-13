import os
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI

load_dotenv()


model = AzureChatOpenAI(
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
)
