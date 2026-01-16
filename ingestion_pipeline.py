import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
print("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))

load_dotenv()

def load_documents(docs_path = "docs"):
    '''load all files from the docs'''
    print(f"Loading Documents from {docs_path}.")
    
    #path exist check
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(path= docs_path,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"The directory {docs_path} does not contain any .txt file.")
    
    for i, doc in enumerate(documents[:3]):# it will show only first 3 documents
        print(f"\nDocument{i+1}: ")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadeta: {doc.metadata}")
        
    return documents

def split_documents(documents, chunk_size = 1000, chunk_overlap=0):
    '''Split documents into smaller chunks with overlap'''
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(chunk_size= chunk_size,
                                          chunk_overlap = chunk_overlap)
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---Chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: ")
            print(chunk.page_content)
            print("-" * 40)
            
        if len(chunks) > 5:
            print(f"\n... and {len(chunks)- 5} more chunks")
            
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    '''create and persist ChromaDB vector store'''
    print("Creating Embeddings and Storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    #create ChromaDB vector store
    print("---Creating Vector store---")
    vectorstore = Chroma.from_documents(
        documents= chunks,
        embedding= embedding_model,
        persist_directory= persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store Created and saved to {persist_directory}")
    return vectorstore

def main():
    '''Main Ingestion Pipeline'''
    print("---RAG Document Ingestion Pipeline---\n")
    
    #Define paths
    docs_path = "docs"
    persistent_directory = "db/chorma_db"
    
    #check if vector store already exists
    if os.path.exists(persistent_directory):
        print("Vector Store Already Exist. no need to Reprocess documents.")
        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory= persistent_directory,
            embedding_function= embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Load existing vector store with {vectorstore._collecton.count()} documents")
        return vectorstore
    
    print("persistent directory does not exist. Initializing vector store...\n")
    
    #load document
    documents = load_documents(docs_path)
    
    #split into chucks
    chunks = split_documents(documents)
    
    #Create Vector Store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore
    
if __name__ == "__main__":
    main()