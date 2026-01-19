from langchain_classic.document_loaders import OnlinePDFLoader, CSVLoader, WebBaseLoader


# PDF
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load()
print(data)

# CSV
loader = CSVLoader(file_path="./articles.csv")
data = loader.load()
print(data)

# Webpage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain")
docs = loader.load()
print(docs)
