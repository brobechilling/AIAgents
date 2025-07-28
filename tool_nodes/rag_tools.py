from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "./vectorstore"
collection_name = "vroomvroom_user_manual"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

try:
    vectorstore = Chroma(
        embedding_function=embeddings,  
        persist_directory=persist_directory,
        collection_name=collection_name
    )
except Exception as e:
    print(f"Error connecting to the ChromaDB: {str(e)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 6} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the VroomVroom User Manual document."""
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the VroomVroom User Manual document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)
