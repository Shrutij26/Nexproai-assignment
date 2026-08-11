import sys
import os
import json
import warnings
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from api import app

client = TestClient(app)

def run_tests():
    print("=== Testing Scenario 1: Question WITH an answer in the context ===")
    # 'lancedb' is present in sample.md and sample.html
    req1 = {
        "question": "Why is LanceDB cost-efficient?",
        "k": 2
    }
    resp1 = client.post("/query", json=req1)
    print(f"Status Code: {resp1.status_code}")
    print(json.dumps(resp1.json(), indent=2))
    
    print("\n=== Testing Scenario 2: Question WITHOUT an answer ===")
    # 'pineapple' is not in the text
    req2 = {
        "question": "What is the capital of France?",
        "k": 2
    }
    resp2 = client.post("/query", json=req2)
    print(f"Status Code: {resp2.status_code}")
    print(json.dumps(resp2.json(), indent=2))
    
    print("\n=== Testing Scenario 3: Metadata-filtered query ===")
    # Search for something but restrict to sample.html where it might/might not exist
    req3 = {
        "question": "Why is LanceDB cost-efficient?",
        "k": 2,
        "filter_key": "filename",
        "filter_value": "sample.html"
    }
    resp3 = client.post("/query", json=req3)
    print(f"Status Code: {resp3.status_code}")
    print(json.dumps(resp3.json(), indent=2))

if __name__ == "__main__":
    run_tests()
