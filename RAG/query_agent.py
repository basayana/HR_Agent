import os
import time
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def main():
    print("Connecting to local LLM and ChromaDB database...")
    
    # 1. Initialize models
    # Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
    Settings.llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # 2. Add a Callback Manager to measure exact timings
    callback_manager = CallbackManager([])
    Settings.callback_manager = callback_manager

    # 3. Point to existing chroma_db directory
    db_path = os.path.join(os.getcwd(), "chroma_db")
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}.")
        return
        
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_collection("hr_markdown_documents")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # Create the engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    print("\n==============================================")
    print("   🤖 Local HR Agent RAG System Active 🤖     ")
    print("==============================================")
    print("Ask a question about HR policies or type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
            
        print("\nProcessing...")
        
        # --- Time the Retrieval Phase ---
        start_retrieve = time.time()
        retriever = index.as_retriever(similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(query)
        end_retrieve = time.time()
        retrieval_time = end_retrieve - start_retrieve
        
        # --- Time the Generation Phase ---
        start_generate = time.time()
        try:
            response = query_engine.query(query)
            end_generate = time.time()
            generation_time = end_generate - start_generate
            
            print(f"\nHR Agent:\n{response}\n")
            print("-" * 50)
            # Print performance metrics
            print(f"⏱️  Retrieval Time (ChromaDB Search): {retrieval_time:.2f} seconds")
            print(f"⏱️  Generation Time (Ollama Llama3.1): {generation_time:.2f} seconds")
            print(f"📦 Total Turnaround Time: {(retrieval_time + generation_time):.2f} seconds")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()
