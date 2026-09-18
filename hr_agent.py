"""A small local HR Q&A agent using TF-IDF retrieval over policy documents."""

from __future__ import annotations

import argparse
from pathlib import Path

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

    def answer(self, question: str) -> tuple[str, str, float]:
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.document_matrix).flatten()
        best_index = scores.argmax()
        confidence = float(scores[best_index])

        if confidence < 0.05:
            return (
                "I couldn't find that in the available leave, WFH, or expense policies. "
                "Please contact HR for guidance.",
                "No matching policy",
                confidence,
            )

        # This concise, extractive generation deliberately stays grounded in policy text.
        context = self.passages[best_index]
        return f"According to the {self.sources[best_index]}: {context}", self.sources[best_index], confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about company HR policies.")
    parser.add_argument(
        "--policies", type=Path, default=Path(__file__).parent / "policies",
        help="Directory containing .txt policy files (default: ./policies)",
    )
    args = parser.parse_args()
    agent = HRPolicyAgent(args.policies)

    print("HR Policy Q&A Agent")
    print("Ask about leave, work from home, or expenses. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not question:
            continue
        answer, source, _confidence = agent.answer(question)
        print(f"\nHR Agent: {answer}\nSource: {source}\n")


if __name__ == "__main__":
    main()
