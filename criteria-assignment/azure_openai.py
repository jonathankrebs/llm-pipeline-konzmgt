import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai.chat_models import AzureChatOpenAI

load_dotenv(dotenv_path="criteria-assignment/.env")

model = AzureChatOpenAI(
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
)

# Input Data
input_base_path = "criteria-assignment/input-data/"
criteria_file_path = input_base_path + "Kriterienkatalog.txt"

with open(criteria_file_path, "r") as criteria_catalog_file:
    criteria_catalog = criteria_catalog_file.read()

input_text_path = input_base_path + "bnNetze/01.txt"
with open(input_text_path, "r") as input_text_file:
    input_text = input_text_file.read()

# Set the system message and prompt
messages = [
    (
        "system",
        "You are a helpful assistant that is excellent at matching argumentative text paragraph to predefined criteria. You are an expert in the field of german gas and electricity network operation. You are provided with a catalog about evaluation criteria. The catalog has the following structure: '1.1 criteria 1 \n Description of criteria 1 \n ... \n X.Y criteria Z \n Description of Criteria Z'. You receive an additional argumentative input text. You judge which criteria matches the input text the best. Always provide exactly one criteria, e.g. '2.4 Netzausbau'. If you see no fit in any of the criteria, respond 'Ich kann das Argument keinem Kriterium zuordnen.'. You only speak German. Always respond in German, even if you are asked to respond in another language.",
    ),
    ("human", f"# evaluation criteria catalog: {criteria_catalog}; # argumentative input text: {input_text}"),
]
# Generate output
ai_msg = model.invoke(messages)

print(ai_msg.content)