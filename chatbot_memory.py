from langchain_openai.chat_models import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_classic.chains.conversation.base import ConversationChain


chat = ChatOpenAI(temperature=0.0)
chat

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("")
    if user_input.lower() == "quit":
        break
    print(conversation.predict(input=user_input))


# Adding buffer to prevent memory from getting large
memory = ConversationBufferWindowMemory(k=2) # Window size = 2
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=False
)

while True:
    user_input = input("")
    if user_input.lower() == "quit":
        break
    print(conversation.predict(input=user_input))
