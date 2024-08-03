from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from gpt4all import GPT4All
import yaml
from datetime import datetime

load_dotenv(dotenv_path="concept-summary/given-text/.env")

model_base_path = "local/models/"
model_name = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
model = GPT4All(
    model_name=model_name, 
    model_path=model_base_path, 
    device="cuda", 
    n_ctx=8192
) 

# Input Data
criteria = ""
input_base_path = "concept-summary/given-text/input-data/datteln/eon/"

criteria_file_path = input_base_path + "criteria.yaml"
with open(criteria_file_path, "r", encoding="utf-8") as f:
    criteria_dict = yaml.safe_load(f)

for criteria, context_files in criteria_dict.items():

    all_documents_content = []

    for context_file in context_files:
        # Parse pdf input
        full_path = f"{input_base_path}{context_file}.pdf"
        loader = PyPDFLoader(full_path)
        pages = loader.load_and_split()
        document_content = ''.join([page.page_content for page in pages])
        all_documents_content.append(document_content)
    # Merge all retrieved document pages into one string    
    combined_documents_content = ''.join(all_documents_content)

    # Set the system message and prompt
    system_message = "You are a helpful assistant that summarizes a given text. You will always be given exactly one important criteria and one input_text to summarize. Summarize only the aspects of the text that relate to this one criteria. Discard all information that doesn't directly relate to the criteria, even if it is included in the input_text. Your summary is used for documentation of contracts. Be very precise and don't use ambiguous phrases. Don't copy unprecise marketing fluff or marketing buzzwords, like 'Wir bieten den besten Support an.' or 'Wir sind hochmodernisiert'. Always use the specific terms of the given text, never use synonyms that don't occur in the given text. Use bullet points to structure your summarization. You only produce German output, no matter the input language. Never answer in any other language than German, even if you're asked to."

    prompt = f"criteria: {criteria}; input_text: {combined_documents_content}"

    # Generate output
    with model.chat_session():
        model.generate(system_message, max_tokens=0)
        ai_msg = model.generate(prompt, max_tokens=4096)

    output_file_path = "concept-summary/given-text/output-data/datteln/eon/llama_3_8b-instruct_q8.txt"
    with open(output_file_path, "a", encoding="utf-8") as output_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_file.write(f"Timestamp: {timestamp}\n")
        output_file.write(f"Criteria: {criteria}\n")
        output_file.write(f"Output:\n{ai_msg}\n")
        output_file.write("---------------------------------------------------------------------\n")

    print(f"The LLM response and criteria have been written to {output_file_path}")
