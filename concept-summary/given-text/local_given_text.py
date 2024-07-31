import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import GPT4All

load_dotenv(dotenv_path="concept-summary/given-text/.env")

model_base_path = "local/models/"
# model_name = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
model_name = "Meta-Llama-3-8B-Instruct-fp16.gguf"
model = GPT4All(
    model=model_base_path + model_name,
    device = "cuda",
    n_threads=8
)

# Input Data
criteria = "Technische Betriebsstelle in Datteln o. Umgebung"
input_base_path = "concept-summary/given-text/input-data/"
input_pdfs = ["09-10.pdf"]

# Parse pdf input
all_documents_content = []
for input_pdf in input_pdfs:
    full_path = input_base_path + input_pdf
    loader = PyPDFLoader(full_path)
    pages = loader.load_and_split()

    document_content = ''.join([page.page_content for page in pages])
    all_documents_content.append(document_content)

combined_documents_content = ''.join(all_documents_content)

# Set the system message and prompt
messages = [
    (
        "system",
        "You are a helpful assistant that summarizes a given text. You will always be given exactly one important criteria and one input_text to summarize. Summarize only the aspects of the text that relate to this one criteria. Discard all information that doesn't directly relate to the criteria, even if it is included in the input_text. Your summary is used for documentation of contracts. Be very precise and don't use ambiguous phrases. Don't copy unprecise marketing fluff or marketing buzzwords, like 'Wir bieten den besten Support an.' or 'Wir sind hochmodernisiert'. Always use the specific terms of the given text, never use synonyms that don't occur in the given text. Use bullet points to structure your summarization. You only produce German output, no matter the input language. Never answer in any other language than German, even if you're asked to.",
    ),
    ("human", f"criteria: {criteria}; input_text: {combined_documents_content}"),
]
# Generate output
ai_msg = model.invoke(messages)
print(ai_msg)
