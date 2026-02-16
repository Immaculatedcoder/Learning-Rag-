# rfs1_ollama_rag.py  ✅ Full working RAG: Local Embeddings + Local LLM (Ollama)

import os
from pathlib import Path

import bs4
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# -----------------------------
# 0) Load .env (optional) + USER_AGENT
# -----------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)  # ok if .env doesn't exist

os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "emmanuel-rag-app")


# -----------------------------
# 1) Load documents from the web
# -----------------------------
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
    header_template={"User-Agent": os.environ["USER_AGENT"]},  # avoids warning in many setups
)
docs = loader.load()


# -----------------------------
# 2) Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# -----------------------------
# 3) Local embeddings + Chroma vector store (no OpenAI needed)
# -----------------------------
# One-time install if needed:
#   pip install -U langchain-huggingface sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="lilianweng_agents_rag",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -----------------------------
# 4) Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the provided context to answer the question. "
     "If the answer is not in the context, say you don't know. Do not make things up."
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# -----------------------------
# 5) Local LLM via Ollama (no OpenAI needed)
# -----------------------------
# Make sure you have pulled a model, e.g.:
#   ollama pull llama3.1
# If you pulled a different model, change model="..." below.
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)


# -----------------------------
# 6) RAG chain
# -----------------------------
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# 7) Ask a question
# -----------------------------
if __name__ == "__main__":
    question = "What is the main idea of ReAct, and how is it different from standard prompting?"
    print("\n--- ANSWER ---\n")
    print(rag_chain.invoke(question))
