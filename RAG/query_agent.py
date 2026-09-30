import os
import time
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding

def retrieve(query, index, top_k=3):
    """
    PHASE 1: THE RETRIEVER
    Searches ChromaDB for relevant document context based on the user query.
    """
    start_time = time.time()
    
    # Initialize a retriever component from our existing index
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return retrieved_nodes, elapsed_time

def generate(query, retrieved_nodes, query_engine):
    """
    PHASE 2: THE GENERATOR
    Takes the query and retrieved context chunks, hands them to the chosen LLM, 
    and synthesizes a natural response.
    """
    start_time = time.time()
    
    # LlamaIndex synthesizes the final answer using the already retrieved content
    response = query_engine.synthesize(query, nodes=retrieved_nodes)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return response, elapsed_time

def main():
    print("==============================================")
    print("      💼 HR Agent Setup Configurator 💼       ")
    print("==============================================")
    print("Choose your LLM generation engine:")
    print("1. Local Ollama (llama3.1:8b) - Private, but slower")
    print("2. Groq Cloud (llama3-8b-8192) - Blazing fast, requires internet")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    # Configure the chosen LLM based on user selection
    if choice == "2":
        if not os.environ.get("GROQ_API_KEY"):
            print("\n❌ Error: GROQ_API_KEY environment variable is not set.")
            print("Please run: $env:GROQ_API_KEY='your_key' in PowerShell first.")
            return
        print("\nConfiguring Groq Cloud LLM...")
        Settings.llm = Groq(model="openai/gpt-oss-20b", request_timeout=60.0)
        mode_label = "⚡ Groq Cloud Mode ⚡"
    else:
        print("\nConfiguring Local Ollama LLM...")
        Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
        mode_label = "🤖 Local Ollama Mode 🤖"

    # Keep embedding layer local regardless of the generation engine chosen
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Point to the existing ChromaDB storage path
    db_path = os.path.join(os.getcwd(), "chroma_db")
    if not os.path.exists(db_path):
        print(f"❌ Error: Database folder not found at {db_path}. Please run creation script first.")
        return
        
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_collection("hr_markdown_documents")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load vector store metadata index into LlamaIndex
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # Setup core query synthesizer engine 
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    print("\n==============================================")
    print(f"   {mode_label}   ")
    print("==============================================")
    print("Ask a question about HR policies or type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
            
        print("\nProcessing RAG pipeline...")
        try:
            # 1. Execute Retrieval Stage
            nodes, t_retrieve = retrieve(query, index, top_k=3)
            
            # 2. Execute Generation Stage
            response, t_generate = generate(query, nodes, query_engine)
            
            # Display localized HR answer
            print(f"\nHR Agent:\n{response}\n")
            
            # Diagnostics & Metrics
            print("-" * 50)
            print(f"⏱️  Retrieval Time (Local ChromaDB): {t_retrieve:.2f} seconds")
            print(f"⏱️  Generation Time (LLM Processing): {t_generate:.2f} seconds")
            print(f"📦 Total Turnaround Time: {(t_retrieve + t_generate):.2f} seconds")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred in the pipeline: {e}\n")

if __name__ == "__main__":
    main()
