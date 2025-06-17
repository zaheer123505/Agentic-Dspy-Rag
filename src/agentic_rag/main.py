# src/agentic_rag/main.py
import os
from dotenv import load_dotenv
# New, explicit, and robust line
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

import dspy
import google.generativeai as genai
from qdrant_client import QdrantClient
from openai import OpenAI
from typing import List, Optional
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Import our custom modules from within the same package
from agentic_rag.components.data_modules import QdrantRetrieverWithExpansion, GeminiReranker
from agentic_rag.components.agents import OrchestratorAgent, SimpleRAG, ComparativeRAG, MultiStepRAG

# --- FASTAPI APP SETUP ---
app = FastAPI(
    title="Professional Agentic RAG API",
    version="5.0",
    description="An advanced RAG system with multiple specialized agents."
)

@app.on_event("startup")
def startup_event():
    """Initializes all models and agents when the server starts."""
    print("--- Server is starting up... ---")
    
    # Load configuration from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not all([openai_api_key, gemini_api_key, qdrant_url, qdrant_collection_name]):
        raise ValueError("One or more environment variables are missing. Check your .env file.")

    # Configure the primary LLM for the application
    llm = dspy.LM(model='gpt-4o', api_key=openai_api_key, max_tokens=1500)
    dspy.settings.configure(lm=llm)
    print("DSPy's LLM configured.")
    
    # Initialize the data modules (retriever and reranker)
    retriever = QdrantRetrieverWithExpansion(
        url=qdrant_url, 
        collection_name=qdrant_collection_name, 
        openai_api_key=openai_api_key
    )
    reranker = GeminiReranker(api_key=gemini_api_key)
    
    # Initialize the specialized "tool" agents
    simple_agent = SimpleRAG(retriever=retriever, reranker=reranker)
    comparative_agent = ComparativeRAG(retriever=retriever, reranker=reranker)
    multi_step_agent = MultiStepRAG(simple_rag_agent=simple_agent)
    
    # Build the main orchestrator agent and store it in the app's state
    app.state.orchestrator = OrchestratorAgent(
        simple_agent=simple_agent, 
        comparative_agent=comparative_agent,
        multi_step_agent=multi_step_agent
    )
    print("Orchestrator Agent initialized and ready.")

# --- API DATA MODELS ---
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

# A module to condense the history and new question
condense_question_with_history = dspy.ChainOfThought("chat_history, new_question -> standalone_question")

# --- API ENDPOINTS ---
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "message": "Welcome to the Professional Agentic RAG API!"}

@app.post("/query", response_model=QueryResponse, summary="Process a Query")
def process_query(request: QueryRequest):
    """
    Accepts a user's question and optional chat history, processes it with the
    full agentic pipeline, and returns a structured answer.
    """
    print(f"\nReceived new question: '{request.question}'")
    
    final_question = request.question
    # If there's chat history, condense the question first
    if request.chat_history:
        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in request.chat_history])
        print("--- Condensing question with chat history ---")
        condensed = condense_question_with_history(chat_history=history_str, new_question=request.question)
        final_question = condensed.standalone_question
        print(f"--- Standalone question: '{final_question}' ---")

    # Call the main orchestrator with the final, context-aware question
    orchestrator = app.state.orchestrator
    prediction = orchestrator(final_question)
    
    # Extract the detected intent from the last trace for the response
    try:
        detected_intent = dspy.settings.lm.history[-1]['response'].intent
    except (IndexError, AttributeError):
        detected_intent = "Unknown"

    return QueryResponse(
        answer=prediction.answer, 
        context=prediction.context,
        intent=detected_intent
    )