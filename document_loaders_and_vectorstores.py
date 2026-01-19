import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.document_loaders import CSVLoader
from langchain_classic.vectorstores import DocArrayInMemorySearch
from langchain_classic.indexes import VectorstoreIndexCreator

# Document loader and vectorstore
file = "articles.csv"
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query = "Do you have any articles on AI?"
response = index.query(query)
print(response)

query = "List all articles you have on AI. Make sure to return them as a numbered list with title and summary."
response = index.query(query)
print(response)

# Similarity Search
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

loaded_articles = loader.load()
loaded_articles[0]

embedded_query = embeddings.embed_query("The Rise of AI Ethics")
embedded_query[:10]

vector_db = DocArrayInMemorySearch.from_documents(
    loaded_articles,
    embeddings
)

title_of_last_article = "The Rise of AI Ethics"

most_similar_articles = vector_db.similarity_search(title_of_last_article, k=6)
most_similar_articles
