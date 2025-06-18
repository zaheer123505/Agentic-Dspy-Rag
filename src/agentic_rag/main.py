# src/agentic_rag/main.py
import os
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))



import dspy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
# Import our upgraded modules
from agentic_rag.components.data_modules import QdrantRetrieverWithExpansion, GeminiReranker
from agentic_rag.components.agents import OrchestratorAgent, SimpleRAG, ComparativeRAG, MultiStepRAG

# --- FASTAPI APP SETUP ---
app = FastAPI(
    title="Robust Agentic RAG API",
    version="6.0",
    description="An advanced RAG system with error handling and usage monitoring."
)

# --- NEW: Global Exception Handler for Quota Errors ---
@app.exception_handler(LiteLLMRateLimitError)
async def litellm_rate_limit_exception_handler(request: Request, exc: LiteLLMRateLimitError):
    """
    Catches the actual RateLimitError from LiteLLM and returns a user-friendly 503 response.
    """
    return JSONResponse(
        status_code=503,
        content={"message": f"API Quota Error: The upstream API provider (e.g., OpenAI) has indicated a rate limit or quota issue. Please check your billing details with the provider. Original error: {exc}"},
    )

@app.on_event("startup")
def startup_event():
    """Initializes all models and agents when the server starts."""
    print("--- Server is starting up... ---")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not all([openai_api_key, gemini_api_key, qdrant_url, qdrant_collection_name]):
        raise ValueError("One or more environment variables are missing.")

    llm = dspy.LM(model='gpt-4o', api_key=openai_api_key, max_tokens=1500)
    dspy.settings.configure(lm=llm)
    print("DSPy's LLM configured.")
    
    retriever = QdrantRetrieverWithExpansion(url=qdrant_url, collection_name=qdrant_collection_name, openai_api_key=openai_api_key)
    reranker = GeminiReranker(api_key=gemini_api_key)
    
    simple_agent = SimpleRAG(retriever=retriever, reranker=reranker)
    comparative_agent = ComparativeRAG(retriever=retriever, reranker=reranker)
    multi_step_agent = MultiStepRAG(simple_rag_agent=simple_agent)
    
    app.state.orchestrator = OrchestratorAgent(
        simple_agent=simple_agent, 
        comparative_agent=comparative_agent,
        multi_step_agent=multi_step_agent
    )
    print("Orchestrator Agent initialized and ready.")

# --- API DATA MODELS and ENDPOINTS  ---
class ChatMessage(BaseModel):
    role: str
    content: str
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []
class QueryResponse(BaseModel):
    answer: str
    context: list[str]
    intent: str
condense_question_with_history = dspy.ChainOfThought("chat_history, new_question -> standalone_question")

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, summary="Process a Query")
def process_query(request: QueryRequest):
    print(f"\nReceived new question: '{request.question}'")
    final_question = request.question
    if request.chat_history:
        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in request.chat_history])
        print("--- Condensing question with chat history ---")
        condensed = condense_question_with_history(chat_history=history_str, new_question=request.question)
        final_question = condensed.standalone_question
        print(f"--- Standalone question: '{final_question}' ---")
    orchestrator = app.state.orchestrator
    prediction = orchestrator(final_question)
    try:
        detected_intent = dspy.settings.lm.history[-1]['response'].intent
    except (IndexError, AttributeError):
        detected_intent = "Unknown"
    return QueryResponse(answer=prediction.answer, context=prediction.context, intent=detected_intent)