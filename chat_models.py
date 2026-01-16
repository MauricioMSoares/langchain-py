# Chat model example

from langchain_openai.chat_models import ChatOpenAI
from langchain.messages import AIMessage, HumanMessage, SystemMessage


chat = ChatOpenAI(temperature=0, max_tokens=100)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Spanish."),
    HumanMessage(content="I love programming.")
]

chat(messages)


# Conversation example

chat2 = ChatOpenAI(temperature=0, max_tokens=100)

context = [SystemMessage(content="You are a friendly chatbot that likes to have conversations.")]

while True:
    user_message = input("You: ")
    context.append(HumanMessage(content=user_message))
    if user_message.lower() == "exit":
        break
    response = chat2(context)

    context.append(response)
    print("AI: ", response.content)
