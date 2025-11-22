from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# 1) Connect Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

# 2) Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3) Load PDFs
loader = DirectoryLoader("pdfs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 4) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunks = splitter.split_documents(docs)

# 5) Embed + Upload
batch = []
for i, c in enumerate(chunks):
    emb = model.encode(c.page_content).tolist()
    vec_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
    batch.append({
        "id": vec_id,
        "values": emb,
        "metadata": {"text": c.page_content}
    })

    # Upload in batches of 100
    if len(batch) == 100:
        index.upsert(batch)
        batch = []

# Upload remaining
if batch:
    index.upsert(batch)

print("✅ DONE! Index successfully uploaded to Pinecone.")
