import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_classic.document_loaders import OnlinePDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load_and_split(text_splitter=recursive_splitter)

db = Chroma.from_documents(data, OpenAIEmbeddings())
print(db._collection.count())

# Maximal Marginal Relevance (MMR)
question = "What does the Encoder portion of a transformer do?"

docs = db.max_marginal_relevance_search(question, k=2, fetch_k=3)

for doc in docs:
    print(doc.page_content)
    print("--------------------")

# Querying using metadata
data_with_metadata = []

for doc in data[:5]:
    doc.metadata["source"] = "https://arxiv.org/pdf/1706.03762.pdf"
    data_with_metadata.append(doc)

data_with_metadata += data[5:]

db.delete()
db = Chroma.from_documents(data_with_metadata, OpenAIEmbeddings())
print(db._collection.count())

docs = db.similarity_search(
    question,
    k=3,
    filter={"source": "https://arxiv.org/pdf/1706.03762.pdf"}
)

for doc in docs:
    print(doc)

# Generating a query with LLM
data_with_metadata = []

for i, doc in enumerate(data):
    if i < 7:
        doc.metadata["source"] = "beginning"
    elif 7 <= i < 15:
        doc.metadata["source"] = "middle"
    else:
        doc.metadata["source"] = "end"

    data_with_metadata.append(doc)

db.delete()
db = Chroma.from_documents(data_with_metadata, OpenAIEmbeddings())
print(db._collection.count())

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="Whether this section of the paper is in the beginning, middle, or end of the paper.",
        type="string"
    )
]

document_content_description = "The forst transformer paper by Google Brain"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "What do they say about the Encoder in the middle of the paper?"
docs = retriever._get_relevant_documents(question)

for doc in docs:
    print(f"SOURCE: {doc.metadata["source"]}")
    print("--------------------")
    print(f"DOCUMENT: {doc.page_content}")
    print("====================")
