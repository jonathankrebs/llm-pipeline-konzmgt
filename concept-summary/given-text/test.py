from langchain_community.llms import GPT4All
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# # There are many CallbackHandlers supported, such as
# # from langchain.callbacks.streamlit import StreamlitCallbackHandler

model = GPT4All(model="concept-summary/given-text/input-data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", device = "cuda", n_threads=8)

# Generate text. Tokens are streamed through the callback manager.
output = model.invoke("List 5 Pokemon.")
print(output)

from gpt4all import GPT4All

# model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="NVIDIA GeForce RTX 4090") #.to(device) # downloads / loads a 4.66GB LLM
# with model.chat_session():
#     print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))

criteria = "Personalsituation im Unternehmen"
combined_documents_content = "Es gibt insgesamt 2503 Facharbeiter."

messages = [
    (
        "system",
        "You are a helpful assistant that summarizes a given text. You will always be given exactly one important criteria and one input_text to summarize. Summarize only the aspects of the text that relate to this one criteria. Discard all information that doesn't directly relate to the criteria, even if it is included in the input_text. Your summary is used for documentation of contracts. Be very precise and don't use ambiguous phrases. Don't copy unprecise marketing fluff or marketing buzzwords, like 'Wir bieten den besten Support an.' or 'Wir sind hochmodernisiert'. Always use the specific terms of the given text, never use synonyms that don't occur in the given text. Use bullet points to structure your summarization. You only produce German output, no matter the input language. Never answer in any other language than German, even if you're asked to.",
    ),
    ("human", f"criteria: {criteria}; input_text: {combined_documents_content}"),
]
# Generate output
# ai_msg = model.invoke(messages)