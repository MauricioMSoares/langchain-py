from langchain_openai.chat_models import ChatOpenAI
from langchain.messages import AIMessage, HumanMessage, SystemMessage


chat = ChatOpenAI(temperature=0, max_tokens=100)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Spanish."),
    HumanMessage(content="I love programming.")
]
chat(messages)
