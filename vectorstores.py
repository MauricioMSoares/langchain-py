import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_classic.document_loaders import OnlinePDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load_and_split(text_splitter=recursive_splitter)
print(data)

db = Chroma.from_documents(data, OpenAIEmbeddings())
print(db._collection.count())

query = "What does the Encoder portion of a transformer do?"
docs = db.similarity_search(query)
print(docs[0].page_content)
