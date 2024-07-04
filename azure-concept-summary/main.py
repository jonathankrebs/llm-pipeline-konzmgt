import os
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
import certifi 

load_dotenv()

model = AzureChatOpenAI(
    azure_deployment="Testbereitstellung-US",
    api_version="2024-05-01-preview"
)

messages = [
    (
        "system",
        "You are a helpful assistant that only produces German output, no matter in which language a question is asked. Never answer in any other language than German, even if you're asked to.",
    ),
    ("human", "What is concession management"),
]
ai_msg = model.invoke(messages)
print(ai_msg.content)
