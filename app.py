from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import traceback

# Load environment variables (only needed locally)
load_dotenv()

# -------------------------
# Local Embedding Model
# -------------------------
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(text):
    return model.encode(text).tolist()


# -------------------------
# Pinecone Setup
# -------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))


# -------------------------
# Gemini LLM Setup
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

Answer respectfully and accurately:
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
# Flask App
# -------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query_api():

    
    try:
        user_input = request.form.get("input_text", "")
        if "created" in user_input.lower() and "sirat" in user_input.lower():
            return jsonify({"response": "SiratGPT was created by Zaid, a visionary AI engineer and entrepreneur who is passionate about fusing technology with knowledge. As the Founder of HatchUp.ai, Zaid built SiratGPT to bring deep Islamic insights to the digital world, combining modern AI techniques with timeless wisdom. His expertise in AI, app development, and automation drives this project, making SiratGPT a unique and intelligent guide for seekers of knowledge"})

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
    port = int(os.getenv("PORT", 7860))
    print(f"🔥 SiratGPT running on port {port}")
    app.run(host="0.0.0.0", port=port)
