import os
os.environ["OPENAI_API_KEY"] = ""
import requests
import xml.etree.ElementTree as ET
from langchain_classic.document_loaders import OnlinePDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import Chroma
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI


def extract_pdf_links(url):
    # Fetch the content of the URL
    response = requests.get(url)
    response.raise_for_status()

    # Parse the XML content
    root = ET.fromstring(response.content)

    # Extract all the links with rdf:resource attribute (using XML namespaces)
    namespaces = {"rdf": "https://www.w3.org/1999/02/22-rdf-syntax-ns#"}
    links = {li.attrib["{%s}resource" % namespaces["rdf"]] for li in root.findall(".//rdf:li", namespaces)}

    # Convert 'abs' links to 'pdf' links
    pdf_links = [link.replace("/abs/", "/pdf/") + ".pdf" for link in links]

    return pdf_links

url = "https://export.arxiv.org/rss/cs.AI"
pdf_links = extract_pdf_links(url)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Load and split data
all_docs = []
for link in pdf_links[:10]:
    print(link)
    loader = OnlinePDFLoader(link)
    data = loader.load_and_split(text_splitter=recursive_splitter)
    all_docs.extend(data)

def estimate_tokens(docs):
    num_est_tokens = 0
    for doc in docs:
        num_est_words = len(doc.page_content.split(" "))
        token_est = num_est_words * 1.333
        num_est_tokens += token_est

    return num_est_tokens

price_per_1000_tokens = .0001
num_estimated_tokens = estimate_tokens(all_docs)
estimated_price = price_per_1000_tokens * (num_estimated_tokens / 1000)
print(num_estimated_tokens)
print(estimated_price)

db = Chroma.from_documents(all_docs, OpenAIEmbeddings())
print(db._collection.count())

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

question = "What is OpenFlamingo?"
result = qa({"question": question})
result["answer"]

question = "What is the link to the GitHub repo?"
result = qa({"question": question})
result["answer"]

while True:
    user_input = input("You: ")

    if user_input == "quit":
        break

    result = qa({"question": user_input})
    print(f"Assistant: {result["answer"]}")
