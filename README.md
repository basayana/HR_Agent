# HR Policy Q&A Agent

A small local Python CLI that answers questions using retrieval-augmented generation (RAG) over plain-text HR policy files. It uses TF-IDF embeddings with cosine similarity to retrieve the best policy paragraph, then returns a grounded answer from that context.

## Setup

With the supplied virtual environment activated:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
.\venv\Scripts\python.exe hr_agent.py 

```

Example questions:

- `How many paid leave days do I receive?`
- `Can I work from home three days each week?`
- `When do I need to submit an expense claim?`

Type `quit` or `exit` to end the session.

## Policies

The policies are individual `.txt` files in `policies/`. Add or edit those files to change the agent's knowledge base; restart the CLI after making changes. You can select another policy directory with `--policies PATH`.

## RAG tracing

The LlamaIndex RAG app sends traces to Arize Phoenix by default. Install the
dependencies from `requirements.txt`, start Phoenix with `phoenix serve`, then
run the RAG CLI or Streamlit app. Open <http://localhost:6006> to inspect
traces. Setup, cloud configuration, and the disable switch are documented in
[RAG/observability/README.md](RAG/observability/README.md).
