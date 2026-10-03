import os
import sys
from rag_engine import HRAgentEngine

def main():
    # Windows consoles may default to cp1252, which cannot print the UI symbols below.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    print("==============================================")
    print("      💼 HR Agent Setup Configurator 💼       ")
    print("==============================================")
    print("Choose your LLM generation engine:")
    print("1. Local Ollama (llama3.1:8b)")
    print("2. Groq Cloud (openai/gpt-oss-20b)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    use_api = (choice == "2")
    
    try:
        # Spin up core RAG pipeline object
        agent = HRAgentEngine(use_api=use_api)
        mode_label = "⚡ Groq Cloud Active ⚡" if use_api else "🤖 Local Ollama Active 🤖"
        print(f"\n==============================================\n   {mode_label}\n==============================================")
    except Exception as e:
        print(f"\n❌ Initialization Failed: {e}")
        return

    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        print("Processing RAG pipeline backend...")
        # Direct call to the unified engine API
        result = agent.ask_question(query)
        
        print(f"\nHR Agent:\n{result['answer']}\n")
        print("-" * 50)
        print(f"⏱️  Retrieval Time:  {result['retrieval_time']:.2f} seconds")
        print(f"⏱️  Generation Time: {result['generation_time']:.2f} seconds")
        print(f"📦 Total Latency:    {result['total_time']:.2f} seconds")
        
        if result['token_metrics']:
            m = result['token_metrics']
            print(f"📊 Question Usage:   {m['current_total']} tokens ({m['current_prompt']} prompt / {m['current_completion']} completion)")
            print(f"📈 Total Consumed:   {m['today_cumulative']} tokens")
            print(f"📅 Remaining Today:  {m['today_remaining']} tokens")
        print("-" * 50)

if __name__ == "__main__":
    main()
