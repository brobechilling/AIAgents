from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

pdf_path = "VroomVroom_User_Manual.pdf"
persist_directory = "./vectorstore"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

collection_name = "vroomvroom_user_manual"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)

try:
    pages = loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=300
)

chunks = splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

print("Vectorstore created and saved to disk.")
