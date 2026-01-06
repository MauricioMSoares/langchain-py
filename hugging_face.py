from getpass import getpass
import os
from langchain_huggingface.llms import HuggingFacePipeline

HUGGING_FACE_HUB_API_TOKEN = getpass()
os.environ["HUGGING_FACE_HUB_API_TOKEN"] = HUGGING_FACE_HUB_API_TOKEN


llm = HuggingFacePipeline(model_id="", model_kwargs={"temperature": 0, "max_length": 64})
text = "Please answer the following question. What is the boiling point of water in Fahrenheit?"
llm(text)
