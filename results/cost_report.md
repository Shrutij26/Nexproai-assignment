# Cost Analysis & Trade-offs Report

## Why the Embedded Approach is Cheaper for Light Workloads
Managed vector databases like Pinecone charge based on "always-on" provisioned infrastructure (pods or compute instances) to keep the index entirely in memory for ultra-low microsecond latency. If an application is lightly queried, this means you are paying full price for idle RAM 99% of the time. 

In contrast, embedded solutions like **LanceDB** store the vector index directly on cheap persistent disk (like AWS EBS or object storage) and run within the application's existing memory space. Because LanceDB is heavily optimized for fast disk-reads rather than relying purely on RAM, the only infrastructure cost is raw storage ($0.08/GB), completely eliminating the idle compute/RAM tax.

## Assumptions Used
1. **Embedding Dimension**: 1536 (OpenAI `text-embedding-3-small`).
2. **Vector Datatype**: `float32` (4 bytes per dimension = 6.144 KB per vector).
3. **Metadata Overhead**: Estimated at 1 KB per vector to store the text chunk, filename, and indexing overhead.
4. **Storage Medium (Embedded)**: AWS EBS gp3 volumes at **$0.08 / GB / month**.
5. **Managed DB Model**: Pinecone Standard Pods. One `p1` pod costs roughly **$70/month** and supports ~1M vectors.

## When a Managed Vector Database Becomes Preferable
While LanceDB is exceptionally cost-efficient, you would switch back to a Managed DB (like Pinecone or Weaviate Cloud) under these conditions:
1. **Massive QPS**: The application scales to thousands of Queries Per Second (QPS), requiring dedicated fleet management and load-balancing that outscales the API server's local resources.
2. **Microsecond Latency Constraints**: If the application requires sub-millisecond retrieval (e.g., real-time recommendation engines), pure in-memory managed databases will outperform disk-backed embedded databases.
3. **Complex Multi-Tenancy**: If the application requires complex Role-Based Access Control (RBAC) at the namespace/tenant level out-of-the-box.
