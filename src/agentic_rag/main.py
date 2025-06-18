# src/agentic_rag/main.py
# This file serves as the main entry point for the FastAPI application.
# It initializes the agentic system and defines the API endpoints for user interaction.

import os
from dotenv import load_dotenv

# --- Environment Configuration ---
# Build a robust path to the .env file at the project root.
# This ensures that environment variables are loaded reliably, regardless of
# where the script is executed from.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

# --- Core Library Imports ---
import dspy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# --- Custom Module Imports ---
# Import the specific error from the underlying 'litellm' library to create a global handler.
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
# Import our custom-built data and agent modules.
from agentic_rag.components.data_modules import QdrantRetrieverWithExpansion, GeminiReranker
from agentic_rag.components.agents import OrchestratorAgent, SimpleRAG, ComparativeRAG, MultiStepRAG

# --- FastAPI Application Setup ---
# Initialize the FastAPI app with metadata for documentation.
app = FastAPI(
    title="Professional Agentic RAG API",
    version="6.1",
    description="An advanced RAG system with error handling and multiple specialized agents."
)

# --- Global Exception Handler ---
@app.exception_handler(LiteLLMRateLimitError)
async def litellm_rate_limit_exception_handler(request: Request, exc: LiteLLMRateLimitError):
    """
    This function acts as a global "safety net".
    It catches RateLimitError exceptions raised by any upstream API (OpenAI, Gemini)
    and returns a user-friendly, structured JSON response with a 503 status code.
    This prevents the server from crashing and provides a clear error message to the client.
    """
    print(f"RATE LIMIT ERROR CAUGHT: {exc}") # Log the specific error for debugging.
    return JSONResponse(
        status_code=503, # 503 Service Unavailable is the standard code for this type of error.
        content={"error": "Service Unavailable", "message": "An upstream API provider has reported a rate limit or quota issue. Please check your API key and billing details."}
    )

# --- Application Startup Logic ---
@app.on_event("startup")
def startup_event():
    """
    This function runs only ONCE when the FastAPI server starts.
    It's responsible for loading all configurations, initializing all models,
    and building the main agentic system. This is highly efficient as the
    heavy models are loaded into memory once and reused for all subsequent requests.
    """
    print("--- Server is starting up... ---")
    
    # Load all necessary credentials and configuration from the environment.
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    # A validation check to ensure the application doesn't start without critical config.
    if not all([openai_api_key, gemini_api_key, qdrant_url, qdrant_collection_name]):
        raise ValueError("One or more environment variables are missing. Check your .env file.")

    # Configure the primary Large Language Model (LLM) for DSPy to use.
    llm = dspy.OpenAI(model='gpt-4o', api_key=openai_api_key, max_tokens=1500)
    dspy.settings.configure(lm=llm)
    print("DSPy's LLM configured.")
    
    # Initialize the data pipeline components (retriever and reranker).
    retriever = QdrantRetrieverWithExpansion(url=qdrant_url, collection_name=qdrant_collection_name, openai_api_key=openai_api_key)
    reranker = GeminiReranker(api_key=gemini_api_key)
    
    # Initialize the specialized "tool" agents, injecting the data components they need.
    simple_agent = SimpleRAG(retriever=retriever, reranker=reranker)
    comparative_agent = ComparativeRAG(retriever=retriever, reranker=reranker)
    multi_step_agent = MultiStepRAG(simple_rag_agent=simple_agent)
    
    # Build the main orchestrator agent, providing it with its toolbox of specialized agents.
    # Store the single, powerful orchestrator instance in the app's state for reuse across all requests.
    app.state.orchestrator = OrchestratorAgent(
        simple_agent=simple_agent, 
        comparative_agent=comparative_agent,
        multi_step_agent=multi_step_agent
    )
    print("Orchestrator Agent initialized and ready.")

# --- API Data Models (Pydantic) ---
# These models define the expected structure for API requests and responses.
# FastAPI uses them for automatic data validation, serialization, and documentation.
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

# A standalone DSPy module for handling conversational memory.
condense_question_with_history = dspy.ChainOfThought("chat_history, new_question -> standalone_question")

# --- API Endpoints ---
@app.get("/", summary="Health Check", tags=["System"])
def read_root():
    """A simple health check endpoint to confirm that the API server is running."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, summary="Process a Query", tags=["Agent"])
def process_query(request: QueryRequest):
    """
    The main endpoint for interacting with the RAG agent.
    It accepts a question and an optional chat history, processes it through
    the full agentic pipeline, and returns a structured answer.
    """
    print(f"\nReceived new question: '{request.question}'")
    
    final_question = request.question
    
    # If chat history is provided, use an LLM to condense it into a standalone question.
    # This resolves ambiguities and makes the agent conversational.
    if request.chat_history:
        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in request.chat_history])
        print("--- Condensing question with chat history ---")
        condensed = condense_question_with_history(chat_history=history_str, new_question=request.question)
        final_question = condensed.standalone_question
        print(f"--- Standalone question: '{final_question}' ---")

    # Call the main orchestrator with the final, context-aware question.
    orchestrator = app.state.orchestrator
    prediction = orchestrator(final_question)
    
    # Extract the detected intent from the prediction object for the final response.
    detected_intent = getattr(prediction, 'intent', 'Unknown')
    
    # Return the final, structured response.
    return QueryResponse(
        answer=prediction.answer, 
        context=prediction.context,
        intent=detected_intent
    )