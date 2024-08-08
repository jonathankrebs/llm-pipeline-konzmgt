import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_ollama import ChatOllama
import yaml

load_dotenv(dotenv_path="criteria-assignment/.env")

model_name = "llama-3_1-8b-q8"
model = ChatOllama(
    model=model_name,
)

# Input Data
input_base_path = "criteria-assignment/input-data/"
criteria_file_path = input_base_path + "Kriterienkatalog.txt"

with open(criteria_file_path, "r", encoding="utf-8") as criteria_catalog_file:
    criteria_catalog = criteria_catalog_file.read()

input_text_path = input_base_path + "bnNetze/input.yaml"
with open(input_text_path, "r", encoding="utf-8") as input_file:
    text_items = yaml.safe_load(input_file)

for text_item in text_items:
    # Set the system message and prompt
    messages = [
        (
            "system",
            "You are a helpful assistant that is excellent at matching an argumentative text paragraph to predefined criteria. You are an expert in the field of german gas and electricity network operation. You are provided with a catalog about evaluation criteria. The catalog has the following structure: '1.1 criteria 1 \n Description of criteria 1 \n ... \n X.Y criteria Z \n Description of Criteria Z'. You receive an additional argumentative input text. You judge which criteria matches the input text the best. Use the criteria description to determine the best fit. Always explain why the criteria matches the input text. Always provide exactly one criteria, e.g. '2.4 Netzausbau' and one explanation. If you see no fit in any of the criteria, respond 'Ich kann das Argument keinem Kriterium zuordnen.'.  You only speak German. Always respond in German, even if you are asked to respond in another language.",
        ),
        ("human", f"# evaluation criteria catalog: {criteria_catalog}; # argumentative input text: {text_item["argument"]}"),
    ]
    # # Alternative: Zuordnung zu mehreren Kriterien erlaubt
    #  messages = [
    #     (
    #         "system",
    #         "You are a helpful assistant that is excellent at matching argumentative text paragraph to predefined criteria. You are an expert in the field of german gas and electricity network operation. You are provided with a catalog about evaluation criteria. The catalog has the following structure: '1.1 criteria 1 \n Description of criteria 1 \n ... \n X.Y criteria Z \n Description of Criteria Z'. You receive an additional argumentative input text. You judge which criteria matches the input text the best. Use the criteria description to determine the best fit. Always explain why the criteria matches the input text. If you see no fit in any of the criteria, respond 'Ich kann das Argument keinem Kriterium zuordnen.'. If some arguments of the input text match criteria A and other arguments of the input text match criteria B, include both criteria in your answer and explain it. You only speak German. Always respond in German, even if you are asked to respond in another language.",
    #     ),
    #     ("human", f"# evaluation criteria catalog: {criteria_catalog}; # argumentative input text: {text_item["argument"]}"),
    # ]
    # Generate output
    ai_msg = model.invoke(messages)

    output_file_path = "criteria-assignment/output-data/bnNetze/llama-3_1-8b-instruct-q8_new.txt"
    with open(output_file_path, "a", encoding="utf-8") as output_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_file.write(f"Timestamp: {timestamp}\n")
        output_file.write(f"Original Criteria: {text_item["original_criteria"]}\n")
        output_file.write(f"Argument: {text_item["argument"]}\n")
        output_file.write(f"Output:\n{ai_msg.content}\n")
        output_file.write("---------------------------------------------------------------------\n")

    print(f"The LLM response and criteria have been written to {output_file_path}")