from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

model = AzureChatOpenAI(
    azure_deployment="GPT-4o-Germany-Central-West",
    api_version="2024-05-01-preview"
)

# Parse pdf input
loader = PyPDFLoader("concept-summary/input-data/EON_Personalsituation_Unternehmen.pdf")
pages = loader.load_and_split()
document_input = ''.join([page.page_content for page in pages])


# Set the system message and prompt
messages = [
    (
        "system",
        "You are a helpful assistant that summarizes a given text. You will always be given exactly one important criteria and one input_text to summarize. Summarize only the aspects of the text that relate to this one criteria. Your summary is used for documentation of contracts. Be very precise and don't use ambiguous phrases. Always use the specific terms of the given text, never use synonyms that don't occur in the given text. Use bullet points to structure your summarization. You only produce German output, no matter the input language. Never answer in any other language than German, even if you're asked to.",
    ),
    (f"human", "criteria: Personalsituation Unternehmen; input_text: {}".format(document_input)),
]

# Generate output
ai_msg = model.invoke(messages)
print(ai_msg.content)
