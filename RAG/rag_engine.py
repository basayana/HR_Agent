import os
import time
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike  # Used for FreeLLMAPI integration
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from token_tracker import update_and_calculate_tokens
from observability.phoenix_tracing import configure_phoenix_tracing

class HRAgentEngine:
    def __init__(self, use_api=True):
        self.use_api = use_api
        configure_phoenix_tracing()
        # Initialize selected LLM
        if self.use_api:
            # Point Settings.llm to your locally running FreeLLMAPI server
            Settings.llm = OpenAILike(
                model="auto", # Change this to the exact model string from your FreeLLMAPI dashboard
                api_key="freellmapi-e198ca14b4b28e505d50662534e3c8de8a0d590a7344ed47", # Paste your FreeLLMAPI Unified API Key here
                api_base="http://localhost:3001/v1",             # The Base URL shown on your FreeLLMAPI keys screen
                is_chat_model=True,                              # Tells LlamaIndex it supports chat completions
                request_timeout=120.0
            )
        else:
            # Fallback to local Ollama if proxy is disabled
            Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

        # Initialize shared embedding layer
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Load Database
        rag_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(rag_dir, "chroma_db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChromaDB not found at {db_path}. Please build the DB first.")
            
        db = chromadb.PersistentClient(path=db_path)
        chroma_collection = db.get_collection("hr_markdown_documents")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    def retrieve(self, query, similarity_top_k=3):
        """Retrieve the most relevant policy nodes without generating an answer."""
        start_retrieve = time.time()
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        retrieved_nodes = retriever.retrieve(query)
        t_retrieve = time.time() - start_retrieve

        return {
            "nodes": retrieved_nodes,
            "retrieval_time": t_retrieve,
        }

    def generate(self, query, retrieved_nodes):
        """Generate an answer from previously retrieved nodes, without retrieving again."""
        start_generate = time.time()
        response = self.query_engine.synthesize(query, nodes=retrieved_nodes)
        t_generate = time.time() - start_generate

        # Disabling or adjusting token metrics logic for the proxy tier if needed
        token_metrics = None
        if not self.use_api:
             token_metrics = update_and_calculate_tokens(response, query, retrieved_nodes)

        return {
            "answer": str(response),
            "generation_time": t_generate,
            "token_metrics": token_metrics,
        }

    def ask_question(self, query):
        """Run retrieval and generation and return the combined pipeline result."""
        retrieval = self.retrieve(query)
        generation = self.generate(query, retrieval["nodes"])

        # Package data into a highly structured interface-agnostic object
        return {
            "answer": generation["answer"],
            "retrieval_time": retrieval["retrieval_time"],
            "generation_time": generation["generation_time"],
            "total_time": retrieval["retrieval_time"] + generation["generation_time"],
            "token_metrics": generation["token_metrics"],
        }
