from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio

# Import your existing RAG modules
from main import main, initialize_llm, get_prompt, get_chain, ask_with_rag

# -----------------------------
# FastAPI app initialization
# -----------------------------
app = FastAPI(
    title="Company Policy AI Assistant",
    description="RAG-powered AI assistant for answering employee queries from company policies",
    version="1.0"
)

from fastapi.middleware.cors import CORSMiddleware

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development allow all origins; restrict in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# Request and Response Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]


# -----------------------------
# Initialize LLM & Chain globally (load once)
# -----------------------------
llm_model = initialize_llm()
prompt_template = get_prompt()
rag_chain = get_chain(llm_model, prompt_template)


# -----------------------------
# Background processing function
# -----------------------------
def process_rag_pipeline(user_query: str):
    """
    Runs RAG retrieval and LLM answer generation.
    """
    retrieved_docs = main(user_query)

    if not retrieved_docs:
        return None, []

    answer = ask_with_rag(rag_chain, query=user_query, retrieved_docs=retrieved_docs)
    docs_preview = [
        doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
        for doc in retrieved_docs
    ]
    return answer, docs_preview


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/ask", response_model=QueryResponse)
async def ask_policy(query_request: QueryRequest):
    """
    Endpoint to ask questions about company policies asynchronously.
    """
    user_query = query_request.query

    # Run the RAG pipeline in a background thread to avoid blocking
    answer, docs_preview = await asyncio.to_thread(process_rag_pipeline, user_query)

    if answer is None:
        raise HTTPException(status_code=404, detail="No relevant documents found for the query.")

    return QueryResponse(answer=answer, retrieved_docs=docs_preview)


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
