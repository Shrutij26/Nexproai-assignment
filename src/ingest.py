import os
import hashlib
from typing import List, Dict, Any
import lancedb
import pyarrow as pa
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load config
from config import Config

def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".html":
        return BSHTMLLoader(file_path, bs_kwargs={'features': 'html.parser'})
    elif ext == ".md":
        # TextLoader works fine for markdown
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def generate_id(text: str, filename: str, chunk_index: int) -> str:
    """Generate a unique deterministic ID based on the chunk content and metadata."""
    content = f"{filename}_{chunk_index}_{text}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def process_documents(data_dir: str) -> List[Dict[str, Any]]:
    print(f"Scanning directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Creating it.")
        os.makedirs(data_dir)
        return []

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    
    # Initialize embeddings model
    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY.startswith("sk-mock") or Config.OPENAI_API_KEY.startswith("your_api") or Config.OPENAI_API_KEY == "your_groq_api_key_here":
        print("Using FakeEmbeddings for testing since no valid API key is provided.")
        from langchain_core.embeddings import FakeEmbeddings
        embeddings_model = FakeEmbeddings(size=384) # Default size for all-MiniLM-L6-v2
    else:
        embeddings_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    
    records = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        print(f"Processing {filename}...")
        try:
            loader = get_loader(file_path)
            docs = loader.load()
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(docs)
            print(f"  -> Generated {len(chunks)} chunks.")
            
            if not chunks:
                continue
                
            # Batch embed all chunks for this file
            texts = [chunk.page_content for chunk in chunks]
            embeddings = embeddings_model.embed_documents(texts)
            
            # Prepare records
            filetype = os.path.splitext(filename)[1].lower()
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                record_id = generate_id(chunk.page_content, filename, i)
                records.append({
                    "id": record_id,
                    "vector": embedding,
                    "text": chunk.page_content,
                    "filename": filename,
                    "filetype": filetype,
                    "chunk_index": i
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return records

def ingest_to_lancedb(records: List[Dict[str, Any]]):
    if not records:
        print("No records to ingest.")
        return
        
    print(f"Connecting to LanceDB at {Config.LANCEDB_URI}")
    db = lancedb.connect(Config.LANCEDB_URI)
    
    # Define schema explicitly to ensure types match
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), len(records[0]["vector"]))),
        pa.field("text", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("filetype", pa.string()),
        pa.field("chunk_index", pa.int32())
    ])
    
    table_name = Config.TABLE_NAME
    
    try:
        tbl = db.open_table(table_name)
        table_exists = True
    except Exception:
        table_exists = False
        
    if not table_exists:
        print(f"Creating new table: {table_name}")
        tbl = db.create_table(table_name, data=records, schema=schema)
        print(f"Ingested {len(records)} records.")
    else:
        print(f"Table {table_name} exists. Performing idempotent merge-insert.")
        # Count before
        count_before = tbl.count_rows()
        print(f"Rows before ingest: {count_before}")
        
        # Idempotent merge: insert if id not exists, update if exists
        tbl.merge_insert("id") \
           .when_matched_update_all() \
           .when_not_matched_insert_all() \
           .execute(records)
           
        # Count after
        count_after = tbl.count_rows()
        print(f"Rows after ingest: {count_after}")
        print(f"Added {count_after - count_before} new records.")
        
    print("\n--- INGESTION SUMMARY ---")
    print(f"Number of documents processed: {len(set(r['filename'] for r in records))}")
    print(f"Number of chunks created: {len(records)}")
    print(f"Number of vectors stored in LanceDB: {tbl.count_rows()}")
    
    sample = tbl.head(1).to_pandas().to_dict(orient='records')[0]
    if 'vector' in sample:
        sample['vector'] = f"<vector of length {len(sample['vector'])}>"
    print(f"Sample stored chunk:\n{sample}")
    print("-------------------------\n")

def main():
    records = process_documents(Config.DATA_DIR)
    ingest_to_lancedb(records)

if __name__ == "__main__":
    main()
