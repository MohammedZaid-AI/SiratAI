from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

# Local embeddings (FREE, offline)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDFs
loader = DirectoryLoader("pdfs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunks = splitter.split_documents(docs)

# Embed & upload
batch = []
for i, c in enumerate(chunks):
    vector = model.encode(c.page_content).tolist()

    vec_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"

    batch.append({
        "id": vec_id,
        "values": vector,
        "metadata": {"text": c.page_content}
    })

    if len(batch) == 100:
        index.upsert(batch)
        batch = []

if batch:
    index.upsert(batch)

print("✅ DONE! Index uploaded to Pinecone.")
