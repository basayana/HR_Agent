"""A small local HR Q&A agent using TF-IDF retrieval over policy documents."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HRPolicyAgent:
    """Retrieves the relevant policy passage and produces a grounded response."""

    def __init__(self, policy_dir: Path) -> None:
        self.passages, self.sources = self._load_passages(policy_dir)
        if not self.passages:
            raise ValueError(f"No .txt policy files found in {policy_dir}")

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.document_matrix = self.vectorizer.fit_transform(self.passages)

    @staticmethod
    def _load_passages(policy_dir: Path) -> tuple[list[str], list[str]]:
        passages: list[str] = []
        sources: list[str] = []
        for policy_file in sorted(policy_dir.glob("*.txt")):
            text = policy_file.read_text(encoding="utf-8").strip()
            paragraphs = text.split("\n\n")
            title = " ".join(paragraphs[0].split())
            # Skip a standalone document heading: it is too broad to be useful
            # retrieval context. Prefix it to each passage instead.
            content_paragraphs = paragraphs[1:] if len(paragraphs) > 1 else paragraphs
            # Paragraphs make retrieval more precise while retaining useful context.
            for paragraph in content_paragraphs:
                cleaned = " ".join(paragraph.split())
                if cleaned:
                    passages.append(f"{title}. {cleaned}")
                    sources.append(policy_file.stem.replace("_", " ").title())
        return passages, sources

    def answer(self, question: str) -> tuple[str, str, str, float]:
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.document_matrix).flatten()
        best_index = scores.argmax()
        confidence = float(scores[best_index])

        if confidence < 0.05:
            return (
                "I couldn't find that in the available leave, WFH, or expense policies. "
                "Please contact HR for guidance.",
                "No matching policy",
                "No matching policy passage",
                confidence,
            )

        context = self.passages[best_index]
        return context, self.sources[best_index], context, confidence


def generate_answer(client: OpenAI, question: str, context: str):
    """Generate an answer and return it with token usage and response headers."""
    raw_response = client.chat.completions.with_raw_response.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using the provided policy context. If the context "
                    "does not contain the answer, say exactly: \"I don't have that "
                    "information in the policy documents\"."
                ),
            },
            {
                "role": "user",
                "content": f"Policy context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    response = raw_response.parse()
    return (
        response.choices[0].message.content
        or "I don't have that information in the policy documents",
        response.usage,
        raw_response.headers,
    )


def header_as_int(headers, name: str) -> int | None:
    """Return an integer HTTP header value, or None when it is unavailable."""
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about company HR policies.")
    parser.add_argument(
        "--policies", type=Path, default=Path(__file__).parent / "policies",
        help="Directory containing .txt policy files (default: ./policies)",
    )
    args = parser.parse_args()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        parser.error("GROQ_API_KEY environment variable is required.")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    agent = HRPolicyAgent(args.policies)

    print("HR Policy Q&A Agent")
    print("Ask about leave, work from home, or expenses. Type 'quit' or 'exit' to stop.\n")
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    token_limit: int | None = None
    remaining_tokens: int | None = None

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"quit", "exit"}:
            print("\nSession token usage:")
            print(f"  Prompt tokens: {total_prompt_tokens}")
            print(f"  Completion tokens: {total_completion_tokens}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Groq token limit: {token_limit if token_limit is not None else 'unavailable'}")
            print(
                "  Groq remaining tokens: "
                f"{remaining_tokens if remaining_tokens is not None else 'unavailable'}"
            )
            if token_limit and remaining_tokens is not None:
                consumed_percent = (token_limit - remaining_tokens) / token_limit * 100
                print(f"  Groq token limit consumed: {consumed_percent:.2f}%")
            else:
                print("  Groq token limit consumed: unavailable")
            print("Goodbye.")
            break
        if not question:
            continue
        retrieved_passage, source, passage, _confidence = agent.answer(question)
        if source == "No matching policy":
            answer = retrieved_passage
        else:
            answer, usage, headers = generate_answer(client, question, retrieved_passage)
            if usage is not None:
                total_prompt_tokens += usage.prompt_tokens or 0
                total_completion_tokens += usage.completion_tokens or 0
                total_tokens += usage.total_tokens or 0
            token_limit = header_as_int(headers, "x-ratelimit-limit-tokens")
            remaining_tokens = header_as_int(headers, "x-ratelimit-remaining-tokens")
        print(f"\nHR Agent: {answer}\nSource: {source}\nSource passage: {passage}\n")



if __name__ == "__main__":
    main()
