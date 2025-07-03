# vector_store.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from financial_data_loader import load_and_chunk_pdfs
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data/10K Reports")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
REPORT_PATH = os.path.join(BASE_DIR, "Reports")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

def initialize_vector_store():
    """Create or load Chroma vector store"""
    if not os.path.exists(CHROMA_PATH):
        print("🆕 Creating new vector store...")
        os.makedirs(CHROMA_PATH, exist_ok=True)
        pdf_chunks = load_and_chunk_pdfs()
        vector_store = Chroma.from_documents(
            documents=pdf_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print(f"💾 Vector store created at {CHROMA_PATH}")
    else:
        print("🔍 Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        print(f"✅ Loaded vector store with {vector_store._collection.count()} documents")
    return vector_store

def get_retriever(company=None, year=None):
    """Create filtered retriever with metadata"""
    filters = {}
    if company:
        filters["company"] = company.lower()
    if year:
        filters["year"] = year
        
    return vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": filters
        }
    )

# Global vector store instance
vector_store = initialize_vector_store()