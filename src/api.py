import os
import time
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import lancedb
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import Config

app = FastAPI(title="Cost-Efficient RAG API")

# Request/Response schemas
class QueryRequest(BaseModel):
    question: str
    k: int = 5
    filter_key: Optional[str] = None
    filter_value: Optional[str] = None

class Chunk(BaseModel):
    id: str
    text: str
    filename: str
    filetype: str
    chunk_index: int
    score: float # distance score

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Chunk]
    metrics: Dict[str, Any]

# Helper to get embeddings model
def get_embeddings_model():
    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY.startswith("sk-mock") or Config.OPENAI_API_KEY.startswith("your_api") or Config.OPENAI_API_KEY == "your_groq_api_key_here":
        from langchain_core.embeddings import FakeEmbeddings
        # all-MiniLM-L6-v2 has dimension 384
        return FakeEmbeddings(size=384)
    return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

def get_llm():
    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY.startswith("sk-mock") or Config.OPENAI_API_KEY.startswith("your_api") or Config.OPENAI_API_KEY == "your_groq_api_key_here":
        return None # Indicate fake LLM needed
    # Use OpenAI-compatible endpoint (Groq) for cost-efficient generation
    return ChatOpenAI(
        model="llama3-8b-8192", 
        temperature=0, 
        api_key=Config.OPENAI_API_KEY, 
        base_url="https://api.groq.com/openai/v1"
    )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    start_time = time.time()
    
    # 1. Generate Embeddings
    embeddings_model = get_embeddings_model()
    try:
        query_vector = embeddings_model.embed_query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    db = lancedb.connect(Config.LANCEDB_URI)
    CACHE_TABLE = "semantic_cache"

    # 1.5 Semantic Cache Check
    try:
        if CACHE_TABLE in db.table_names():
            cache_tbl = db.open_table(CACHE_TABLE)
            cache_results = cache_tbl.search(query_vector).limit(1).to_pandas()
            
            if not cache_results.empty:
                best_match = cache_results.iloc[0]
                # If distance is very small, it means someone asked almost the exact same question
                if best_match.get("_distance", 1.0) < 0.15:
                    cache_latency = time.time() - start_time
                    return QueryResponse(
                        answer=best_match["answer"],
                        retrieved_chunks=[],
                        metrics={
                            "retrieved_chunk_count": 0,
                            "retrieval_latency_ms": 0, # Skipped main retrieval
                            "total_latency_ms": round(cache_latency * 1000, 2),
                            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "note": "Served from Semantic Cache (Zero Cost)"}
                        }
                    )
    except Exception as e:
        print(f"Cache read error: {e}")

    # 2. Retrieval
    retrieval_start = time.time()
    db = lancedb.connect(Config.LANCEDB_URI)
    
    try:
        tbl = db.open_table(Config.TABLE_NAME)
    except Exception:
        raise HTTPException(status_code=404, detail="Vector index not found. Please run ingestion first.")

    search = tbl.search(query_vector).limit(request.k)
    
    # Apply metadata filter if provided
    if request.filter_key and request.filter_value:
        search = search.where(f"{request.filter_key} = '{request.filter_value}'")
        
    try:
        results = search.to_pandas()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")
        
    retrieval_latency = time.time() - retrieval_start
    
    # Parse results
    retrieved_chunks = []
    contexts = []
    if not results.empty:
        for idx, row in results.iterrows():
            retrieved_chunks.append(Chunk(
                id=row["id"],
                text=row["text"],
                filename=row["filename"],
                filetype=row["filetype"],
                chunk_index=row["chunk_index"],
                score=row.get("_distance", 0.0)
            ))
            contexts.append(f"[Doc: {row['filename']}, Chunk: {row['chunk_index']}]\n{row['text']}")
            
    context_str = "\n\n".join(contexts)
    
    # 3. LLM Generation
    system_prompt = (
        "You are an AI assistant designed to answer questions strictly based on the provided context.\n"
        "Instructions:\n"
        "1. Use ONLY the retrieved context below to answer the user's question.\n"
        "2. If the context does not contain the answer, you MUST reply exactly with: 'No relevant context found'. Do not hallucinate or guess.\n"
        "3. When you use information from the context, cite your sources by appending the document and chunk reference, e.g., '[Doc: filename.pdf, Chunk: 0]'.\n\n"
        f"Context:\n{context_str}"
    )
    
    llm = get_llm()
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    if llm is None:
        # Fake LLM response for testing without API key
        if not retrieved_chunks:
            answer = "No relevant context found"
        else:
            # Simple heuristic for testing: check for meaningful words
            q_words = set(w for w in request.question.lower().split() if len(w) > 4)
            c_words = set(context_str.lower().split())
            if len(q_words.intersection(c_words)) > 0:
                doc_cite = f"[Doc: {retrieved_chunks[0].filename}, Chunk: {retrieved_chunks[0].chunk_index}]"
                answer = f"This is a mocked answer based on the context. {doc_cite}"
            else:
                answer = "No relevant context found"
    else:
        # Real LLM call
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.question)
        ]
        response = llm.invoke(messages)
        answer = response.content
        if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            token_usage = response.response_metadata["token_usage"]

    # Save successful answers to the Semantic Cache
    if "No relevant context found" not in answer:
        cache_data = [{"vector": query_vector, "question": request.question, "answer": f"⚡ [CACHED] {answer}"}]
        try:
            if CACHE_TABLE in db.table_names():
                cache_tbl = db.open_table(CACHE_TABLE)
                cache_tbl.add(cache_data)
            else:
                db.create_table(CACHE_TABLE, data=cache_data)
        except Exception as e:
            print(f"Cache write error: {e}")

    total_latency = time.time() - start_time
    
    return QueryResponse(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        metrics={
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
            "total_latency_ms": round(total_latency * 1000, 2),
            "token_usage": token_usage
        }
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document and triggers idempotent ingestion."""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    file_path = os.path.join(Config.DATA_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Import ingestion logic directly
        from ingest import process_documents, ingest_to_lancedb
        records = process_documents(Config.DATA_DIR)
        ingest_to_lancedb(records)
        
        return {"message": f"Successfully uploaded and ingested {file.filename}", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
