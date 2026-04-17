from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import uvicorn
import os
from openai import OpenAI

app = FastAPI(title="Pegasus CS Assistant")

# Add CORS middleware - This is the fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # For now allow all (we can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load semantic index
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(
    persist_directory="./wp_semantic_index",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Grok API Client
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.environ.get("GROK_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    docs = retriever.invoke(request.message)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source_url") for doc in docs if doc.metadata.get("source_url")]

    system_prompt = """You are a helpful assistant for Pegasus Communication Solutions.
Answer based only on the provided context from the company's knowledge base.
Be professional, clear, and friendly."""

    response = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.message}"}
        ],
        temperature=0.7,
        max_tokens=800
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources[:5]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))