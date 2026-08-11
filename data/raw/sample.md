# Cost-Efficient RAG Application

A Retrieval-Augmented Generation (RAG) system uses a vector database to fetch relevant context before passing it to an LLM.

## Why LanceDB?
LanceDB is an embedded database that relies heavily on disk storage rather than in-memory storage, making it cost-efficient for large, lightly queried datasets.

## Chunking
Chunking text is essential so that we don't pass the entire document into the LLM context window. Small, overlapping chunks usually provide better retrieval metrics like Recall@K.
