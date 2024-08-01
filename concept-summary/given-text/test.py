from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="NVIDIA GeForce RTX 4090", n_ctx=8192) # downloads / loads a 4.66GB LLM

with model.chat_session():
    model.generate("You are an expert ruby programmer giving advice", max_tokens=0)
    print(model.generate("How to write a loop?"))
