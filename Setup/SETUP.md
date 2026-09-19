# HR Q&A Agent — Setup Guide

A small local HR policy Q&A agent built using [Codex CLI](https://github.com/openai/codex) (OpenAI's command-line coding agent), TF-IDF retrieval, and LLM-based answer generation via [Groq](https://console.groq.com) (free tier, OpenAI-compatible API).

This guide documents the exact steps used to set up and build this project on **Windows**, including the real-world snags encountered and how to fix them.

---

## 1. Prerequisites

| Tool | Purpose | Link |
|---|---|---|
| Node.js (LTS) | Required to install Codex CLI via npm | https://nodejs.org |
| Python 3.11–3.13 | Runs the agent itself | https://www.python.org/downloads |
| VS Code (optional) | Convenient editor + integrated terminal | https://code.visualstudio.com |
| A ChatGPT account | Free sign-in for Codex CLI (no separate API key needed for Codex itself) | https://chat.openai.com |
| A Groq account | Free API key for LLM generation step | https://console.groq.com |

Verify installs:
```powershell
node --version
python --version
```

---

## 2. Install Codex CLI

```powershell
npm install -g @openai/codex
```

> ⚠️ **Package name matters.** Install `@openai/codex` exactly. A plain `codex` package on npm is an unrelated, unmaintained project.

Verify:
```powershell
codex --version
```

Sign in:
```powershell
codex
```
Choose **"Sign in with ChatGPT"** — this uses your existing ChatGPT plan's usage allowance. No separate billing or API key is required for Codex CLI itself.

> Note: Signing in to Codex CLI only authorizes the *Codex tool*. It does **not** give your own code (the agent you build) access to any OpenAI API — that's a separate key, covered in step 5.

---

## 3. Create the project and a virtual environment

```powershell
mkdir HR_Agent
cd HR_Agent
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution-policy error:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Your prompt should now show `(venv)` at the start of the line.

---

## 4. Build the agent with Codex

With the venv active and inside the `HR_Agent` folder, run:
```powershell
codex
```

You'll be asked to trust the directory — choose **"Yes, continue"** (safe, since this is your own empty project folder).

Prompt Codex with something like:
> Build a simple HR Q&A agent in Python. I already have a virtual environment activated at ./venv — install all dependencies there. It should answer employee questions about a small set of HR policies (leave policy, WFH policy, expense policy) stored as text files. Use a simple RAG approach: load the policy docs, do basic keyword/embedding retrieval to find the most relevant policy text for a question, then generate an answer using that context. Include a few sample policy documents and a CLI interface to ask questions interactively.

Codex will generate:
- `hr_agent.py` — main script
- `policies/*.txt` — sample HR policy documents
- `requirements.txt` — dependencies
- `README.md` — its own usage notes

Approve its file-write and command prompts as it works.

### Known retrieval limitation (TF-IDF)
The initial build used **TF-IDF + cosine similarity** for retrieval (not neural embeddings). This is lightweight and keyword-based, but it can favor longer, keyword-dense paragraphs over short, direct-answer sentences. For example, a one-line fact ("Total leave days per year: 31") can lose to a longer, related paragraph in scoring.

**Fix:** rewrite short factual lines as fuller sentences that repeat key terms, e.g.:
> "Total paid leave days per year: employees are entitled to 31 leave days in total per calendar year, combining all leave types."

This gives short facts enough keyword density to be retrieved correctly.

---

## 5. Add real LLM-based generation (Groq, free tier)

The initial build was **extractive** — it returned the retrieved passage directly with no LLM involved. To make it a true generative RAG pipeline:

### 5.1 Get a free Groq API key
1. Sign up at https://console.groq.com (no credit card required).
2. Create an API key from the dashboard.

> Groq was chosen because it offers a genuinely free tier (no card required) serving open-source models (Llama, GPT-OSS, Qwen) at very fast inference speed.

### 5.2 Set the key as an environment variable
Never hardcode API keys in source files. Set it per session:
```powershell
$env:GROQ_API_KEY="your-key-here"
```

### 5.3 Check which models your key can access
Model availability can vary by account. Confirm directly:
```powershell
(Invoke-WebRequest -Uri "https://api.groq.com/openai/v1/models" -Headers @{ Authorization = "Bearer $env:GROQ_API_KEY" } -UseBasicParsing | ConvertFrom-Json).data.id
```

Use one of the **chat/instruct-capable** models returned (e.g. `openai/gpt-oss-20b`). Ignore models meant for other purposes (`whisper-*` = speech-to-text, `*-prompt-guard-*` = safety classifiers, `orpheus-*` = text-to-speech).

### 5.4 Prompt Codex to wire up generation
> Update hr_agent.py to add a real LLM generation step using the Groq API, which is OpenAI-compatible. Use the `openai` Python package but point the base_url to Groq's endpoint (`https://api.groq.com/openai/v1`), and read the API key from the environment variable `GROQ_API_KEY` — never hardcode it. Use a chat-capable model available on the account. Keep the existing TF-IDF retrieval logic exactly as-is. After retrieval finds the best-matching passage, send the retrieved passage plus the user's question to the LLM with a system prompt instructing it to answer only using the provided policy context and to say "I don't have that information in the policy documents" if the context doesn't contain the answer. Print both the LLM-generated answer and the source passage it was grounded in, so I can compare them. Add `openai` to requirements.txt and install it in ./venv.

### 5.5 Run it
```powershell
.\venv\Scripts\python.exe hr_agent.py
```

---

## 6. Common issues encountered (and fixes)

| Issue | Cause | Fix |
|---|---|---|
| Antivirus blocks `CODEX.EXE` ("Application Lock Down") | AV flags an unsigned/new executable making file/network calls | Whitelist the exact path under NPAV/AV exceptions, or check with IT if on a managed machine |
| `openai.NotFoundError: model does not exist` | Model name not available on your specific Groq account | Query `/v1/models` with your key to get the real list, use one of those exactly |
| Edits to a policy file don't seem to apply | File not saved in the editor (unsaved dot indicator) | `Ctrl+S` before re-running |
| `curl -H "..."` fails in PowerShell | PowerShell's `curl` alias is `Invoke-WebRequest`, which needs headers as a hashtable, not a string | Use `-Headers @{ Authorization = "Bearer $env:GROQ_API_KEY" }` |
| Script exits immediately at "You:" prompt | Blank input / accidental Ctrl+C or Ctrl+Z triggers EOF handling | Re-run and type a real question |

---

## 7. Design notes / things worth knowing

- **Retrieval is local; generation is cloud.** The TF-IDF vector index and similarity search run entirely on your machine. Only the final retrieved passage + question are sent to Groq's API for answer generation.
- **No API key set up? The LLM is simply never called** for out-of-scope questions — a hardcoded fallback message is returned instead when retrieval confidence is too low. This avoids hallucinated answers on questions the policy documents don't cover.
- **ChatGPT sign-in ≠ OpenAI API access.** Signing in to Codex CLI with a ChatGPT account only authorizes Codex itself; it does not grant your own code any API access. A separate API key (here, from Groq) is required for the agent's own LLM calls.
- **Security:** never commit your API key to GitHub. Keep it in an environment variable only, and add `.env` (if you introduce one later) to `.gitignore`.

---

## 8. Next step: Evaluation

This agent is the *subject* to be evaluated, not the end deliverable. The next phase uses [DeepEval](https://deepeval.com) to score:
- **Answer Relevancy**
- **Faithfulness / Groundedness**
- **Contextual Precision / Recall**

against a set of test questions and expected answers.
