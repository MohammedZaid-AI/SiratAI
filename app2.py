from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import traceback

load_dotenv()

# -------------------------
# PINECONE SETUP
# -------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

# -------------------------
# EMBEDDING MODEL
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# RETRIEVER FUNCTION
# -------------------------
def retrieve_from_pinecone(query, top_k=3):
    q_emb = embedding_model.encode(query).tolist()

    result = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in result["matches"]]


# -------------------------
# LLM SETUP
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9,
    max_output_tokens=1000
)

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are SiratGPT, an Islamic knowledge assistant.
Use only the given Quran context.

Context:
{context}

User Question:
{query}

Answer respectfully:
"""
)

chain = prompt | llm


# -------------------------
# FLASK SETUP
# -------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query_api():
    try:
        user_input = request.form.get("input_text", "")

        # Easter egg
        if "created" in user_input.lower() and "sirat" in user_input.lower():
            return jsonify({"response":
                "SiratGPT was created by Zaid, Founder of HatchUp.ai, to bring Islamic knowledge to the digital world."
            })

        # Retrieve context
        results = retrieve_from_pinecone(user_input)
        context = "\n\n".join(results)

        if not context.strip():
            return jsonify({"response": "No relevant Quran context found."})

        # LLM response
        response = chain.invoke({"context": context, "query": user_input})
        return jsonify({"response": response.content})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🔥 SiratGPT running on port {port}")
    app.run(host="0.0.0.0", port=port)
