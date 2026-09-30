"""Evaluate RAG retrieval against data/golden_retrieval.jsonl.

Run from this directory with: python test_retrieval.py
Optionally set the number of retrieved nodes: python test_retrieval.py --top-k 5
"""

import argparse
import json
from pathlib import Path

from rag_engine import HRAgentEngine


def source_name(result_node):
    """Get the source filename from a LlamaIndex retrieval result."""
    node = getattr(result_node, "node", result_node)
    metadata = getattr(node, "metadata", {}) or {}
    for key in ("file_name", "file_path", "source"):
        value = metadata.get(key)
        if value:
            return Path(str(value)).name
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of documents to retrieve per query (default: 3)")
    parser.add_argument(
        "--golden-file",
        type=Path,
        default=Path(__file__).parent / "data" / "golden_retrieval.jsonl",
        help="path to the golden JSONL file",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not args.golden_file.is_file():
        parser.error(f"golden file not found: {args.golden_file}")

    cases = [
        json.loads(line)
        for line in args.golden_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    engine = HRAgentEngine()

    full_coverage = 0
    in_scope = 0
    out_of_scope_correct = 0
    precision_sum = 0.0
    recall_sum = 0.0
    true_positive_total = 0
    retrieved_total = 0
    relevant_total = 0
    print(f"Golden set: {args.golden_file}")
    print(f"Retrieving top {args.top_k} nodes per query\n")

    for case in cases:
        expected = set(case.get("relevant_documents", []))
        result = engine.retrieve(case["query"], similarity_top_k=args.top_k)
        found = {source_name(node) for node in result["nodes"]}
        missing = expected - found

        if expected:
            in_scope += 1
            true_positives = len(expected & found)
            precision = true_positives / len(found) if found else 0.0
            recall = true_positives / len(expected)
            precision_sum += precision
            recall_sum += recall
            true_positive_total += true_positives
            retrieved_total += len(found)
            relevant_total += len(expected)

            passed = not missing
            full_coverage += int(passed)
            status = f"P={precision:.1%} R={recall:.1%}"
            if missing:
                status += " MISS " + ", ".join(sorted(missing))
        else:
            passed = not found
            out_of_scope_correct += int(passed)
            status = "PASS (no documents)" if passed else "OUT-OF-SCOPE HIT"

        print(f"{case['id']}: {status}")
        print(f"  retrieved: {', '.join(sorted(found)) if found else '(none)'}")

    total = len(cases)
    coverage = full_coverage / in_scope if in_scope else 0
    macro_precision = precision_sum / in_scope if in_scope else 0
    macro_recall = recall_sum / in_scope if in_scope else 0
    micro_precision = (true_positive_total / retrieved_total
                      if retrieved_total else 0)
    micro_recall = (true_positive_total / relevant_total
                    if relevant_total else 0)
    print("\nSummary")
    print(f"  Macro precision (in-scope): {macro_precision:.1%}")
    print(f"  Macro recall (in-scope):    {macro_recall:.1%}")
    print(f"  Micro precision (in-scope): {micro_precision:.1%}")
    print(f"  Micro recall (in-scope):    {micro_recall:.1%}")
    print(f"  In-scope document coverage: {full_coverage}/{in_scope} ({coverage:.1%})")
    print(f"  Out-of-scope queries with no results: {out_of_scope_correct}/"
          f"{total - in_scope}")
if __name__ == "__main__":
    main()
