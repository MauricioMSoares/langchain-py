import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_classic.document_loaders import WikipediaLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import Chroma
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


docs = WikipediaLoader(query="Langchain", load_max_docs=1).load()
len(docs)

# Build splitter
chunk_size = 2000
chunk_overlap = 200

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)

topics = ["Langchain", "OpenAI", "Generative AI", "Large Language Models", "Natural Language Processing"]

all_docs = []
for topic in topics:
    loader = WikipediaLoader(query=topic, load_max_docs=1)
    docs = loader.load_and_split(recursive_splitter)
    all_docs.extend(docs)

len(all_docs)

# Store data in db
db = Chroma.from_documents(all_docs, OpenAIEmbeddings())
print(db._collection.count())

# Set-up LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever = db.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

while True:
    question = input("User: ")

    if question.lower() == "quit":
        break

    result = qa({"question": question})

    print(f"Assistant: {result["answer"]}")
