import os
import time
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding
from token_tracker import update_and_calculate_tokens

class HRAgentEngine:
    def __init__(self, use_groq=False):
        self.use_groq = use_groq
        
        # Initialize selected LLM
        if self.use_groq:
            if not os.environ.get("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY environment variable is not configured.")
            Settings.llm = Groq(model="openai/gpt-oss-20b", request_timeout=60.0)
        else:
            Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

        # Initialize shared embedding layer
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Load Database
        db_path = os.path.join(os.getcwd(), "chroma_db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChromaDB not found at {db_path}. Please build the DB first.")
            
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_collection("hr_markdown_documents")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    def ask_question(self, query):
        """Processes the full RAG workflow and returns a clean dictionary object."""
        # 1. Retrieval Phase
        start_retrieve = time.time()
        retriever = self.index.as_retriever(similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(query)
        t_retrieve = time.time() - start_retrieve

        # 2. Generation Phase
        start_generate = time.time()
        response = self.query_engine.synthesize(query, nodes=retrieved_nodes)
        t_generate = time.time() - start_generate

        # 3. Token Analytics Phase (Only calculate if Groq backend is active)
        token_metrics = None
        if self.use_groq:
            token_metrics = update_and_calculate_tokens(response, query, retrieved_nodes)

        # Package data into a highly structured interface-agnostic object
        return {
            "answer": str(response),
            "retrieval_time": t_retrieve,
            "generation_time": t_generate,
            "total_time": t_retrieve + t_generate,
            "token_metrics": token_metrics
        }
