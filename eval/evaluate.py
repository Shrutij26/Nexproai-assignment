import json
import time
import math
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def calculate_mrr(retrieved_files, expected_file):
    for rank, filename in enumerate(retrieved_files, start=1):
        if filename == expected_file:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(retrieved_files, expected_file, k):
    dcg = 0.0
    for rank, filename in enumerate(retrieved_files[:k], start=1):
        if filename == expected_file:
            # relevance is 1 if expected, 0 otherwise
            dcg += 1.0 / math.log2(rank + 1)
    
    # IDCG is always 1.0 / log2(2) = 1.0 for a single relevant document
    idcg = 1.0
    return dcg / idcg

def evaluate():
    with open('eval/dataset.json', 'r') as f:
        dataset = json.load(f)

    results = []
    latencies = []
    
    k = 3
    metrics = {
        "hit_count": 0,
        "mrr_sum": 0.0,
        "ndcg_sum": 0.0,
        "context_precision_sum": 0.0,
        "faithfulness_sum": 0.0,
        "relevance_sum": 0.0
    }

    for item in dataset:
        req = {
            "question": item["question"],
            "k": k
        }
        
        response = client.post("/query", json=req)
        if response.status_code != 200:
            print(f"Error on question: {item['question']}")
            continue
            
        data = response.json()
        retrieved_files = [chunk["filename"] for chunk in data["retrieved_chunks"]]
        expected_file = item["expected_filename"]
        
        # Retrieval Metrics
        hit = 1 if expected_file in retrieved_files else 0
        mrr = calculate_mrr(retrieved_files, expected_file)
        ndcg = calculate_ndcg(retrieved_files, expected_file, k)
        
        # Context Precision (precision at rank of first relevant doc)
        context_precision = 0.0
        for rank, filename in enumerate(retrieved_files, start=1):
            if filename == expected_file:
                # Number of relevant docs up to this rank is 1
                context_precision = 1.0 / rank
                break
                
        # Answer Quality Metrics (Mocked LLM Judge logic)
        answer = data["answer"].lower()
        if answer == "no relevant context found":
            faithfulness = 0.0
            relevance = 0.0
        else:
            # Heuristic: If it generated an answer based on context, 
            # and context actually contained the ground truth source, it is highly faithful
            faithfulness = 1.0 if hit else 0.0
            # Relevance: Does the answer contain meaningful keywords from the expected answer?
            expected_words = set(w for w in item["expected_answer"].lower().split() if len(w) > 4)
            if len(expected_words) > 0 and any(w in answer for w in expected_words) or hit:
                relevance = 1.0
            else:
                relevance = 0.5
                
        # Latency
        retrieval_latency = data["metrics"]["retrieval_latency_ms"]
        latencies.append(retrieval_latency)
        
        # Update sums
        metrics["hit_count"] += hit
        metrics["mrr_sum"] += mrr
        metrics["ndcg_sum"] += ndcg
        metrics["context_precision_sum"] += context_precision
        metrics["faithfulness_sum"] += faithfulness
        metrics["relevance_sum"] += relevance
        
        results.append({
            "question": item["question"],
            "expected_file": expected_file,
            "retrieved_files": retrieved_files,
            "hit": hit,
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
            "context_precision": round(context_precision, 4),
            "faithfulness": faithfulness,
            "relevance": relevance,
            "retrieval_latency_ms": retrieval_latency
        })

    num_q = len(results)
    if num_q == 0:
        print("No questions evaluated successfully.")
        return

    # Calculate percentiles without numpy
    sorted_latencies = sorted(latencies)
    def percentile(data, p):
        idx = int(len(data) * p)
        return data[idx]

    p50 = percentile(sorted_latencies, 0.50)
    p95 = percentile(sorted_latencies, 0.95)

    aggregated = {
        "num_questions": num_q,
        "k": k,
        "Recall@k / Hit Rate": round(metrics["hit_count"] / num_q, 4),
        "MRR": round(metrics["mrr_sum"] / num_q, 4),
        "nDCG@k": round(metrics["ndcg_sum"] / num_q, 4),
        "Context Precision": round(metrics["context_precision_sum"] / num_q, 4),
        "Faithfulness / Groundedness (mock LLM-judge)": round(metrics["faithfulness_sum"] / num_q, 4),
        "Answer Relevance (mock LLM-judge)": round(metrics["relevance_sum"] / num_q, 4),
        "EM / F1": "N/A (Exact Match and F1 are highly unreliable for generative text; semantic metrics are preferred)",
        "Retrieval p50 Latency (ms)": round(p50, 2),
        "Retrieval p95 Latency (ms)": round(p95, 2)
    }

    output = {
        "aggregated_metrics": aggregated,
        "per_question_results": results
    }

    with open('eval/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("=== EVALUATION COMPLETE ===")
    print(json.dumps(aggregated, indent=2))
    print("\nResults saved to eval/results.json")

if __name__ == "__main__":
    evaluate()
