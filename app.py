from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import traceback

load_dotenv()

# -------------------------
# HuggingFace: Free Embeddings
# -------------------------
HF_TOKEN = os.getenv("HF_API_KEY")

def embed_query(text):
    url = "https://router.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    response = requests.post(url, headers=headers, json=payload, timeout=20)
    data = response.json()

    if isinstance(data, list) and isinstance(data[0], list):
        return data[0]   # embedding vector
    else:
        raise Exception(f"HF API Error: {data}")



# -------------------------
# Pinecone Setup
# -------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))


# -------------------------
# Gemini LLM
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_output_tokens=700
)

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are SiratGPT, an Islamic knowledge assistant.
Use ONLY the provided Quran context.

Context:
{context}

User Question:
{query}

Answer respectfully and authentically:
"""
)

chain = prompt | llm


# -------------------------
# Retrieval
# -------------------------
def retrieve_from_pinecone(query, top_k=3):
    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    return [m["metadata"]["text"] for m in result["matches"]]


# -------------------------
# Flask Setup
# -------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query_api():
    try:
        user_input = request.form.get("input_text", "")

        results = retrieve_from_pinecone(user_input)
        context = "\n\n".join(results)

        if not context.strip():
            return jsonify({"response": "No relevant Quran context found."})

        response = chain.invoke({"context": context, "query": user_input})
        return jsonify({"response": response.content})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🔥 SiratGPT running on port {port}")
    app.run(host="0.0.0.0", port=port)
