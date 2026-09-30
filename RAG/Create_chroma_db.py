import os
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def main():
    print("Initializing local LLM and Embedding models...")
    
    # 1. Configure Settings with your downloaded llama3.1 model
    Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # Chunk configuration optimized for Markdown files
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # 2. Set up the local ChromaDB storage path
    db_path = os.path.join(os.getcwd(), "chroma_db")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("hr_markdown_documents")
    
    # 3. Connect Chroma to LlamaIndex vector store framework
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Read your Markdown (.md) documents
    data_path = os.path.join(os.getcwd(), "data/md")
    print(f"Reading Markdown files from: {data_path}")
    
    # SimpleDirectoryReader automatically detects and processes .md files natively
    documents = SimpleDirectoryReader(
        input_dir=data_path, 
        required_exts=[".md"]
    ).load_data()

    if not documents:
        print("Warning: No .md files found in the 'data' directory. Please drop some files there first!")
        return

    # 5. Build the index and save it to ChromaDB
    print(f"Processing {len(documents)} document pages/sections into ChromaDB...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    # 6. Create a query engine to interact with your data
    query_engine = index.as_query_engine()
    
    print("\n--- HR Agent RAG System Ready ---")
    while True:
        query = input("\nAsk the HR Agent a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        print("Searching local ChromaDB and generating response...")
        response = query_engine.query(query)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    # Pulls the embedding model automatically if you haven't downloaded it yet
    print("Ensuring embedding model is available locally...")
    os.system("ollama pull nomic-embed-text") 
    main()
