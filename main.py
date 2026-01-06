import os
from langchain_openai.llms import OpenAI
os.environ["OPENAI_API_KEY"] = ""


llm = OpenAI(temperature=0.9)
text = "What is a good name for a framework that makes large language models easier to work with?"
print(llm(text))
