import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def percentile(data, p):
    idx = int(len(data) * p)
    return sorted(data)[idx]

def run_latency_test():
    with open('eval/dataset.json', 'r') as f:
        dataset = json.load(f)

    retrieval_latencies = []
    total_latencies = []

    # Run 5 iterations of the dataset to get a solid sample (16 * 5 = 80 queries)
    iterations = 5
    for i in range(iterations):
        for item in dataset:
            req = {"question": item["question"], "k": 3}
            response = client.post("/query", json=req)
            if response.status_code == 200:
                data = response.json()
                retrieval_latencies.append(data["metrics"]["retrieval_latency_ms"])
                total_latencies.append(data["metrics"]["total_latency_ms"])

    if not retrieval_latencies:
        print("No latency data collected.")
        return

    retrieval_p50 = percentile(retrieval_latencies, 0.50)
    retrieval_p95 = percentile(retrieval_latencies, 0.95)
    
    total_p50 = percentile(total_latencies, 0.50)
    total_p95 = percentile(total_latencies, 0.95)

    results = {
        "num_requests": len(retrieval_latencies),
        "retrieval_latency_ms": {
            "p50": round(retrieval_p50, 2),
            "p95": round(retrieval_p95, 2)
        },
        "end_to_end_latency_ms": {
            "p50": round(total_p50, 2),
            "p95": round(total_p95, 2)
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/latency_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_latency_test()
