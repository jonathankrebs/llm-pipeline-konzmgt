import os
from dotenv import load_dotenv
import yaml
from datetime import datetime

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings  
from langchain_openai.chat_models import AzureChatOpenAI

load_dotenv(dotenv_path="concept-summary/rag/.env")

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

# Init vector store
index_name: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

criteria_file_path = "concept-summary/rag/input-data/datteln/criteria.yaml"
with open(criteria_file_path, "r", encoding="utf-8") as f:
    criteria_dict = yaml.safe_load(f)

# Iterate through all categories and criteria
for category, criteria_list in criteria_dict.items():
    for criteria in criteria_list:
        query = f"Was steht in dem Kapitel '{category}', zu dem Kriterium '{criteria}'?"
        # Perform a similarity search
        docs = vector_store.similarity_search(
            query=query,
            k=3,
            search_type="similarity",
        )
        llm_context = ""
        for doc in docs:
            llm_context += f"{doc.page_content}"


        # LLM generation
        gpt_deployment_name = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT")
        model = AzureChatOpenAI(
            azure_deployment = gpt_deployment_name,
            api_version = azure_openai_api_version
        )

        # Set the system message and prompt
        messages = [
            (
                "system",
                "You are a helpful assistant that summarizes a given text. You are given exactly one important category and criteria and one input text to summarize. Summarize only the aspects of the input text that relate to this one category and criteria. Discard all information that doesn't directly relate to the criteria, even if it is included in the input_text. Your summary is used for documentation of contracts. Be very precise and don't use ambiguous phrases. Don't copy unprecise marketing fluff or marketing buzzwords, like 'Wir bieten den besten Support an.' or 'Wir sind hochmodernisiert'. Always use the specific terms of the input text, never use synonyms that don't occur in the input text. Use bullet points to structure your summarization. You only produce German output, no matter the input language. Never answer in any other language than German, even if you're asked to. Always state at the end of each bullet point which page number the information is obtained from. If you can't identify the correct page number, write 'Seite (?)' instead.",
            ),
            ("human", f"category: {category}, criteria: {criteria}; input text: {llm_context}"),
        ]
        # Generate output
        ai_msg = model.invoke(messages)
        output_file_path = "concept-summary/rag/output-data/datteln/innogy/azure_v03.pdf"
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_file.write(f"Timestamp: {timestamp}\n")
            output_file.write(f"Category: {category}\n")
            output_file.write(f"Criteria: {criteria}\n")
            output_file.write(f"Output:\n{ai_msg.content}\n")
            output_file.write("---------------------------------------------------------------------\n")

        print(f"The LLM response and criteria have been written to {output_file_path}")
