import os
from langchain.document_loaders import AsyncHtmlLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Load webpages
urls = ["https://www.reuters.com/news/archive/technologyNews",
        "https://techcrunch.com/"]

loader = AsyncHtmlLoader(urls)
docs = loader.load()

# Extract content
all_text = []
for doc in docs:
    all_text.append(doc.page_content)

# Split content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0
)

all_splits = []
for text in all_text:
    splits = splitter.create_documents([text])
    all_splits.extend(splits)

# Embed splits
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cuda"}
)

# Store embeddings
vectorstore = FAISS.from_documents(all_splits, embeddings)

# Create chain
chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    return_source_documents=True
)

# Query
chat_history = []
query = "Show me top 10 hot news?"
result = chain({"question": query, "chat_history": chat_history})

print(result["answer"])