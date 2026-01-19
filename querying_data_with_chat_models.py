import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_classic.document_loaders import OnlinePDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load_and_split(text_splitter=recursive_splitter)

db = Chroma.from_documents(data, OpenAIEmbeddings())
print(db._collection.count())

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

question = "What does the Encoder portion of the transformer do?"
result = qa_chain({"query": question})
result

# Adding memory to the model
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

result = qa({"question": question})
result["answer"]

question = "How does the multi-head self-attention mechanism work?"
result = qa({"question": question})
result["answer"]

question = "On which dimension are they concatenated?"
result = qa({"question": question})
result["answer"]

question = "Do you mean they are concatenated on the columns?"
result = qa({"question": question})
result["answer"]
