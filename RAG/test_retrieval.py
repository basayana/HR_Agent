"""Evaluate retrieval with DeepEval contextual precision and recall.

Run from this directory with: python test_retrieval.py
Optionally set the number of retrieved nodes: python test_retrieval.py --top-k 5
"""

import argparse
import json
import os
import asyncio
from pathlib import Path

import pandas as pd
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel
from groq import Groq

from rag_engine import HRAgentEngine


class GroqJudge(DeepEvalBaseLLM):
    """Use Groq's hosted API as DeepEval's LLM judge."""

    def __init__(self, model="qwen/qwen3.8-27b"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not configured.")
        self.model = model
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def get_model_name(self):
        return f"Groq {self.model}"

    def generate(self, prompt: str, schema: BaseModel = None):
        # Format the user instruction block cleanly for JSON schemas
        if schema is not None:
            schema_json = (schema.model_json_schema() if hasattr(schema, "model_json_schema")
                           else schema.schema())
            prompt += (
                "\n\nYou MUST respond with a valid JSON object matching the following structure schema. "
                "Do NOT wrap it in markdown code blocks or backticks. Return ONLY the raw JSON string text:\n" 
                + json.dumps(schema_json)
            )

        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": f"You are a strict data validation agent. Always respond with pure, valid JSON objects. Here is your evaluation task:\n{prompt}"
                }
            ],
            "temperature": 0,
        }
        
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
            
        response = self.load_model().chat.completions.create(**request)
        
        # ⚡ FIX 1: Safely index the choice array from the live network response stream
        output = response.choices[0].message.content or ""
        
        if schema is None:
            return output

        # Parse string block safely to JSON
        parsed = json.loads(output.strip())
        if hasattr(schema, "model_validate"):
            return schema.model_validate(parsed)
        return schema.parse_obj(parsed)

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return await asyncio.to_thread(self.generate, prompt, schema)


def node_text(result_node):
    """Return the text of a LlamaIndex retrieval result."""
    node = getattr(result_node, "node", result_node)
    if hasattr(node, "get_content"):
        return node.get_content()
    return str(getattr(node, "text", node))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of nodes to retrieve per query (default: 3)")
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
    missing_evidence = [case.get("id", "unknown") for case in cases
                        if case.get("relevant_documents") and not case.get("evidence")]
    if missing_evidence:
        parser.error("in-scope golden cases need an 'evidence' value: "
                     + ", ".join(missing_evidence))

    engine = HRAgentEngine()
    judge = GroqJudge()
    precision_metric = ContextualPrecisionMetric(model=judge)
    recall_metric = ContextualRecallMetric(model=judge)
    rows = []

    print(f"Golden set: {args.golden_file}")
    print(f"Retrieving top {args.top_k} nodes per query\n")

    for case in cases:
        # Fetch actual answer context
        retrieved = engine.ask_question(case["query"])
        
        # Extract source matching context nodes cleanly for evaluation
        if hasattr(engine, "index"):
            retrieved_raw = engine.index.as_retriever(similarity_top_k=args.top_k).retrieve(case["query"])
            contexts = [node_text(node) for node in retrieved_raw]
        else:
            contexts = []

        relevant_documents = case.get("relevant_documents", [])

        if relevant_documents:
            test_case = LLMTestCase(
                input=case["query"],
                actual_output=retrieved.get("answer", ""),
                expected_output=case["evidence"],
                retrieval_context=contexts,
            )
            precision_metric.measure(test_case)
            recall_metric.measure(test_case)
            precision = precision_metric.score
            recall = recall_metric.score
            p_reason = precision_metric.reason or ""
            r_reason = recall_metric.reason or ""
        else:
            precision = recall = None
            p_reason = r_reason = ""

        rows.append({
            "id": case.get("id", ""),
            "precision": precision,
            "recall": recall,
            "retrieved_nodes": len(contexts),
            "precision_reason": p_reason,
            "recall_reason": r_reason,
        })
        if precision is None:
            status = "N/A (no reference evidence)"
        else:
            status = f"P={precision:.1%} R={recall:.1%}"
        print(f"{case.get('id', 'unknown')}: {status}")

    results = pd.DataFrame(rows)
    scored = results.dropna(subset=["precision", "recall"])
    print("\nSummary (DeepEval contextual metrics)")
    if scored.empty:
        print("  No in-scope cases to score.")
    else:
        print(f"  Mean contextual precision: {scored['precision'].mean():.1%}")
        print(f"  Mean contextual recall:    {scored['recall'].mean():.1%}")
        print(f"  Scored cases: {len(scored)}/{len(results)}")


if __name__ == "__main__":
    main()
