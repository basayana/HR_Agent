import os
import time
import json
import chromadb
import requests
from datetime import datetime
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding

# --- CONFIGURATION: DAILY CAPS ---
# Set this to match your daily limit from ://groq.com (Default Free Tier is 500,000)
DAILY_TOKEN_LIMIT = 500000 
TOKEN_COUNT_FILE = "groq_token_count.txt"

def update_and_calculate_tokens(response, query, nodes):
    """
    Extracts the exact tokens used for the current request, updates the 
    persistent groq_token_count.txt file, and returns tracking metrics.
    """
    # 1. Extract exact hardware tokens from Groq API response
    p_tokens, c_tokens, q_total = 0, 0, 0
    try:
        if hasattr(response, "additional_kwargs") and "raw" in response.additional_kwargs:
            raw_res = response.additional_kwargs["raw"]
            if hasattr(raw_res, "usage") and raw_res.usage:
                p_tokens = getattr(raw_res.usage, "prompt_tokens", 0)
                c_tokens = getattr(raw_res.usage, "completion_tokens", 0)
                q_total = getattr(raw_res.usage, "total_tokens", 0)
    except Exception:
        pass
        
    # Fallback approximation if metadata extraction gets stripped by wrapper updates
    if q_total == 0:
        p_tokens = int((len(query) + sum([len(n.text) for n in nodes])) / 4)
        c_tokens = int(len(str(response)) / 4)
        q_total = p_tokens + c_tokens

    # 2. Read existing tracking data from the log file
    today_str = datetime.today().strftime('%Y-%m-%d')
    cumulative_tokens = 0
    
    if os.path.exists(TOKEN_COUNT_FILE):
        try:
            with open(TOKEN_COUNT_FILE, 'r') as f:
                data = json.load(f)
                # If the log is from today, keep accumulating. If old date, reset to 0.
                if data.get("date") == today_str:
                    cumulative_tokens = data.get("total_tokens_consumed", 0)
        except Exception:
            pass # Reset/overwrite if file corrupt or empty

    # 3. Add current question tokens to historical total and write back to file
    cumulative_tokens += q_total
    try:
        with open(TOKEN_COUNT_FILE, 'w') as f:
            json.dump({"date": today_str, "total_tokens_consumed": cumulative_tokens}, f)
    except Exception as e:
        print(f"⚠️ Warning: Could not write token data to {TOKEN_COUNT_FILE}: {e}")

    remaining_tokens = max(0, DAILY_TOKEN_LIMIT - cumulative_tokens)
    return p_tokens, c_tokens, q_total, cumulative_tokens, remaining_tokens

def retrieve(query, index, top_k=3):
    """PHASE 1: THE RETRIEVER (Searches ChromaDB via local embedding model)"""
    start_time = time.time()
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)
    return retrieved_nodes, (time.time() - start_time)

def generate(query, retrieved_nodes, query_engine):
    """PHASE 2: THE GENERATOR (Generates text using local Ollama or Groq Cloud)"""
    start_time = time.time()
    response = query_engine.synthesize(query, nodes=retrieved_nodes)
    return response, (time.time() - start_time)

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
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
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
            # 1. Execute Retrieval Stage (Runs Locally)
            nodes, t_retrieve = retrieve(query, index, top_k=3)
            
            # 2. Execute Generation Stage
            response, t_generate = generate(query, nodes, query_engine)
            
            # Display localized HR answer
            print(f"\nHR Agent:\n{response}\n")
            
            # Diagnostics & Performance Metrics
            print("-" * 50)
            print(f"⏱️  Retrieval Time (Local ChromaDB): {t_retrieve:.2f} seconds")
            print(f"⏱️  Generation Time (LLM Processing): {t_generate:.2f} seconds")
            print(f"📦 Total Turnaround Time: {(t_retrieve + t_generate):.2f} seconds")
            
            # --- TOKEN METRICS DASHBOARD FROM PERSISTENT FILE ---
            if choice == "2":
                p, c, current_q, total_day, remaining_day = update_and_calculate_tokens(response, query, nodes)
                print(f"📊 Current Question Usage:       {current_q} tokens ({p} prompt / {c} completion)")
                print(f"📈 Total Consumed Today (Saved): {total_day} tokens")
                print(f"📅 Calculated Remaining for Day: {remaining_day} tokens")
                
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred in the pipeline: {e}\n")

if __name__ == "__main__":
    main()
