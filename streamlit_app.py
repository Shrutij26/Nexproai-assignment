import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

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

# --- HEADER ---
st.title("🔍 Cost-Efficient Document Search")
st.markdown("##### A lightweight RAG pipeline built with LanceDB and FastAPI")
st.info("Hi there! I built this project to show how to run a Retrieval-Augmented Generation (RAG) system without the high infrastructure costs of managed vector databases. By using an embedded LanceDB database on local disk, it brings the 'always-on' cost down to practically zero.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Under the Hood")
    st.markdown(f"**Database:** LanceDB (Embedded)")
    st.markdown(f"**Embeddings:** {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
    st.markdown(f"**LLM:** gpt-4o-mini")
    st.divider()
    
    st.header("📁 Add a Document")
    st.caption("Got a PDF, Markdown, or HTML file? Drop it here to index it instantly.")
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "md", "html"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Upload to Knowledge Base", use_container_width=True):
            with st.status("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success(f"Added {uploaded_file.name}!")
                    else:
                        st.error(f"Something went wrong: {res.text}")
                except Exception as e:
                    st.error(f"Couldn't connect to the backend: {e}")

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
                    response = requests.post(f"{API_URL}/query", json={"question": question, "k": k_value})
                    if response.status_code == 200:
                        data = response.json()
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
                                
                    else:
                        st.error(f"Backend returned an error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Couldn't connect to the backend server. Make sure FastAPI is running.")

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
