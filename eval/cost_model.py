import os
import csv
import json

def run_cost_model():
    # Assumptions
    dim = 1536  # text-embedding-3-small dimension
    bytes_per_dim = 4  # float32
    metadata_overhead_bytes = 1024  # approx 1KB for text, filename, etc.
    
    bytes_per_vector = (dim * bytes_per_dim) + metadata_overhead_bytes
    gb_per_vector = bytes_per_vector / (1024**3)
    
    # AWS EBS gp3 storage cost ($/GB/month)
    ebs_cost_per_gb = 0.08
    
    # Pinecone standard pod pricing (p1 pod supports ~1M vectors, costs ~$70/month)
    pinecone_pod_cost = 70.0
    vectors_per_pod = 1_000_000
    
    scales = [
        {"label": "100K", "count": 100_000},
        {"label": "1M", "count": 1_000_000},
        {"label": "10M", "count": 10_000_000},
    ]
    
    results = []
    
    for scale in scales:
        count = scale["count"]
        storage_gb = count * gb_per_vector
        lancedb_cost = storage_gb * ebs_cost_per_gb
        
        # Managed DB needs minimum 1 pod
        pods_needed = max(1, count / vectors_per_pod)
        managed_cost = pods_needed * pinecone_pod_cost
        
        savings = managed_cost - lancedb_cost
        
        results.append({
            "Vector count": scale["label"],
            "LanceDB estimated storage/cost ($/mo)": f"${lancedb_cost:.2f} ({storage_gb:.2f} GB)",
            "Managed vector DB estimated cost ($/mo)": f"${managed_cost:.2f} ({pods_needed} pod(s))",
            "Difference/savings ($/mo)": f"${savings:.2f}"
        })
        
    os.makedirs('results', exist_ok=True)
    
    # Write to CSV
    csv_path = 'results/cost_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    # Write assumptions to JSON for easy reading
    assumptions = {
        "embedding_dimension": dim,
        "vector_datatype": "float32 (4 bytes)",
        "metadata_overhead": "1 KB per vector",
        "storage_medium": "AWS EBS gp3 ($0.08 / GB / month)",
        "managed_db_model": "Pinecone Standard Pod ($70 / pod / month, 1M capacity per pod)"
    }
    with open('results/assumptions.json', 'w') as f:
        json.dump(assumptions, f, indent=2)
        
    print(json.dumps(results, indent=2))
    print(json.dumps(assumptions, indent=2))

if __name__ == "__main__":
    run_cost_model()
