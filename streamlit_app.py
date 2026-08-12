import streamlit as st
import json
import os
import sys
import time
from dotenv import load_dotenv

# Add src to path so we can import internal modules directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from config import Config
from ingest import process_documents, ingest_to_lancedb
from api import get_embeddings_model, get_llm
from langchain_core.messages import SystemMessage, HumanMessage
import lancedb

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Cost-Efficient RAG", page_icon="🔍", layout="wide")

# Custom CSS for a clean, simple look
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .big-font {
        font-size: 18px !important;
        font-weight: 500;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def local_query(question: str, k: int):
    start_time = time.time()
    embeddings_model = get_embeddings_model()
    try:
        query_vector = embeddings_model.embed_query(question)
    except Exception as e:
        return {"error": f"Embedding error: {str(e)}"}

    db = lancedb.connect(Config.LANCEDB_URI)
    CACHE_TABLE = "semantic_cache"

    try:
        if CACHE_TABLE in db.table_names():
            cache_tbl = db.open_table(CACHE_TABLE)
            cache_results = cache_tbl.search(query_vector).limit(1).to_pandas()
            if not cache_results.empty:
                best_match = cache_results.iloc[0]
                if best_match.get("_distance", 1.0) < 0.15:
                    cache_latency = time.time() - start_time
                    return {
                        "answer": best_match["answer"],
                        "retrieved_chunks": [],
                        "metrics": {
                            "retrieved_chunk_count": 0,
                            "retrieval_latency_ms": 0,
                            "total_latency_ms": round(cache_latency * 1000, 2)
                        }
                    }
    except Exception:
        pass # Ignore cache read errors

    retrieval_start = time.time()
    try:
        tbl = db.open_table(Config.TABLE_NAME)
    except Exception:
        return {"error": "Vector index not found. Please run ingestion first."}

    search = tbl.search(query_vector).limit(k)
    try:
        results = search.to_pandas()
    except Exception as e:
        return {"error": f"Search error: {str(e)}"}

    retrieval_latency = time.time() - retrieval_start
    retrieved_chunks = []
    contexts = []
    if not results.empty:
        for idx, row in results.iterrows():
            retrieved_chunks.append({
                "id": row["id"],
                "text": row["text"],
                "filename": row["filename"],
                "filetype": row["filetype"],
                "chunk_index": row["chunk_index"],
                "score": row.get("_distance", 0.0)
            })
            contexts.append(f"[Doc: {row['filename']}, Chunk: {row['chunk_index']}]\n{row['text']}")

    context_str = "\n\n".join(contexts)

    system_prompt = (
        "You are an AI assistant designed to answer questions strictly based on the provided context.\n"
        "Instructions:\n"
        "1. Use ONLY the retrieved context below to answer the user's question.\n"
        "2. If the context does not contain the answer, you MUST reply exactly with: 'No relevant context found'. Do not hallucinate or guess.\n"
        "3. When you use information from the context, cite your sources by appending the document and chunk reference, e.g., '[Doc: filename.pdf, Chunk: 0]'.\n\n"
        f"Context:\n{context_str}"
    )

    llm = get_llm()
    if llm is None:
        if not retrieved_chunks:
            answer = "No relevant context found"
        else:
            q_words = set(w for w in question.lower().split() if len(w) > 4)
            c_words = set(context_str.lower().split())
            if len(q_words.intersection(c_words)) > 0:
                doc_cite = f"[Doc: {retrieved_chunks[0]['filename']}, Chunk: {retrieved_chunks[0]['chunk_index']}]"
                answer = f"This is a mocked answer based on the context. {doc_cite}"
            else:
                answer = "No relevant context found"
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        answer = response.content

    if "No relevant context found" not in answer:
        cache_data = [{"vector": query_vector, "question": question, "answer": f"⚡ [CACHED] {answer}"}]
        try:
            if CACHE_TABLE in db.table_names():
                cache_tbl = db.open_table(CACHE_TABLE)
                cache_tbl.add(cache_data)
            else:
                db.create_table(CACHE_TABLE, data=cache_data)
        except Exception:
            pass

    total_latency = time.time() - start_time
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "metrics": {
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
            "total_latency_ms": round(total_latency * 1000, 2)
        }
    }

# --- HEADER ---
st.title("🔍 Cost-Efficient Document Search")
st.markdown("##### A lightweight RAG pipeline built with LanceDB and FastAPI")
st.info("Hi there! I built this project to show how to run a Retrieval-Augmented Generation (RAG) system without the high infrastructure costs of managed vector databases. By using an embedded LanceDB database on local disk, it brings the 'always-on' cost down to practically zero.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Under the Hood")
    st.markdown(f"**Database:** LanceDB (Embedded)")
    st.markdown(f"**Embeddings:** {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
    st.markdown(f"**LLM:** Llama-3-8B (via Groq API) *(Note: You can use a real OpenAI API Key for even better results!)*")
    st.divider()
    
    st.header("📁 Add a Document")
    st.caption("Got a PDF, Markdown, or HTML file? Drop it here to index it instantly.")
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "md", "html"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Upload to Knowledge Base", use_container_width=True):
            with st.status("Processing..."):
                try:
                    os.makedirs(Config.DATA_DIR, exist_ok=True)
                    file_path = os.path.join(Config.DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                        
                    records = process_documents(Config.DATA_DIR)
                    ingest_to_lancedb(records)
                    st.success(f"Added {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"Failed to process file: {e}")

    st.divider()
    st.header("📊 Pipeline Performance")
    st.caption("Results from the offline evaluation script.")
    eval_path = "eval/results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
            agg = eval_data.get("aggregated_metrics", {})
            st.metric("Hit Rate (Recall)", f"{agg.get('Recall@k / Hit Rate', 0)*100:.1f}%")
            st.metric("Context Precision", agg.get("Context Precision", "N/A"))
            st.metric("Faithfulness", f"{agg.get('Faithfulness / Groundedness (mock LLM-judge)', 0)*100:.1f}%")
    else:
        st.info("Eval results not found. Run the eval script first.")

# --- TABS ---
tab_query, tab_kb, tab_arch = st.tabs(["Search", "Indexed Documents", "How it Works"])

with tab_query:
    st.markdown("<p class='big-font'>Try it out</p>", unsafe_allow_html=True)
    st.write("You can type your own question below, or click one of the examples to test it out.")
    
    def set_question(q):
        st.session_state.my_question = q
        
    col_a, col_b, col_c = st.columns(3)
    col_a.button("Example: Why is LanceDB cheap?", on_click=set_question, args=("Why is LanceDB cost-efficient?",))
    col_b.button("Example: What is RAG?", on_click=set_question, args=("What does a RAG system use to fetch relevant context?",))
    col_c.button("Example: Unrelated Question", on_click=set_question, args=("What is the capital of France?",))

    st.divider()
    
    q_col, k_col = st.columns([4, 1])
    with q_col:
        question = st.text_input("Ask a question based on the indexed documents:", key="my_question")
    with k_col:
        k_value = st.slider("Docs to retrieve (Top-K)", min_value=1, max_value=10, value=3)

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Looking through documents..."):
                try:
                    data = local_query(question, k_value)
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        answer = data.get("answer", "")
                        chunks = data.get("retrieved_chunks", [])
                        metrics = data.get("metrics", {})
                        
                        st.markdown("---")
                        
                        if "no relevant context found" in answer.lower():
                            st.warning("I couldn't find an answer to that in the uploaded documents.")
                            st.caption("Note: Rather than guessing or making things up, the system is designed to safely say 'I don't know' when the documents don't contain the answer. This is a deliberate safeguard against AI hallucinations.")
                        else:
                            st.success(answer)
                            citations = set([f"📄 {c['filename']}" for c in chunks])
                            st.markdown("**Sources used:** " + ", ".join(citations))

                        st.markdown("---")
                        st.write("#### Search Stats")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Database Search Time", f"{metrics.get('retrieval_latency_ms', 0)} ms")
                        m2.metric("Total API Time", f"{metrics.get('total_latency_ms', 0)} ms")
                        m3.metric("Chunks Read", metrics.get("retrieved_chunk_count", 0))
                        
                        with st.expander("See the exact text retrieved from the database"):
                            for c in chunks:
                                st.markdown(f"**From `{c['filename']}`** (Similarity: `{c['score']:.2f}`)")
                                st.info(c['text'])
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with tab_kb:
    st.write("Here are the raw files that are currently chunked and indexed in the LanceDB database.")
    data_dir = os.getenv("DATA_DIR", "./data/raw")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        if files:
            for f in files:
                with st.expander(f"📄 {f}"):
                    try:
                        with open(os.path.join(data_dir, f), "r", encoding="utf-8") as doc_f:
                            content = doc_f.read(1500)
                            st.text(content + ("\n\n[Content truncated...]" if len(content) == 1500 else ""))
                    except Exception:
                        st.write("This is a PDF or binary file, so the raw text is hidden for readability (but it is fully indexed).")
        else:
            st.info("No documents found.")

with tab_arch:
    st.write("#### Why did I build it this way?")
    st.markdown("""
    When building AI applications, it's very easy to accidentally rack up massive cloud bills. I built this project to prove that you can build a highly accurate, enterprise-grade AI search tool without burning money on unnecessary infrastructure. 
    
    Here are the core design decisions I made to keep costs low and quality high:

    **1. Ditching expensive cloud databases for LanceDB**  
    Most vector databases (like Pinecone) charge you hundreds of dollars a month just to keep your data sitting in memory (RAM) 24/7. That makes sense if you have millions of users querying it every single second. But for internal company tools or lighter workloads, it's a massive waste of money. Instead, I used **LanceDB**. It embeds directly into the app and reads data straight from the hard drive. It is incredibly fast, but brings the database bill down to literally pennies a month.

    **2. Preventing "Duplicate Data" bugs**  
    If a user accidentally uploads the same PDF twice, a poorly written app will index it twice. This ruins the search results and doubles the storage costs. To fix this, I made the ingestion pipeline *idempotent*. I wrote a hashing function that gives every single paragraph a unique fingerprint. If you upload the same document again, the database recognizes the fingerprints and perfectly ignores the duplicates.

    **3. Forcing the AI to tell the truth (Grounding)**  
    Generative AI loves to make things up (hallucinate) when it doesn't know the answer. To solve this, I wrote a strict backend prompt that acts as a leash. It forces the AI to *only* look at the exact text retrieved from our database. If the answer isn't in those documents, the system refuses to answer. This is why you see the polite "I don't know" message instead of a dangerous, hallucinated guess.
    """)
