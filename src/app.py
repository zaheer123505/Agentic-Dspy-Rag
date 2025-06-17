# src/app.py
# FINAL VERSION: Advanced Agentic RAG API with Query Expansion and Chat History.

import os
from dotenv import load_dotenv
load_dotenv()

import dspy
import google.generativeai as genai
from qdrant_client import QdrantClient
from openai import OpenAI
from typing import List, Optional
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# --- Reusable Custom Modules ---

# NEW: Retriever with built-in Query Expansion/Rephrasing
class QdrantRetrieverWithExpansion(dspy.Retrieve):
    def __init__(self, collection_name: str, url: str, openai_api_key: str, k: int = 20):
        super().__init__(k=k)
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-large"
        # A simple module to generate multiple queries
        self.query_rephraser = dspy.ChainOfThought("question -> rephrased_queries", n=1)

    def forward(self, query_or_queries, k: Optional[int] = None):
        k = k if k is not None else self.k
        
        # 1. Rephrase the original query into multiple versions
        rephrase_prompt = f"Generate 3 diverse but related questions to the original query. Keep the core intent. Separate them with a semi-colon. Original Query: {query_or_queries}"
        rephrased = self.query_rephraser(question=rephrase_prompt)
        # Combine the original query with the rephrased ones
        queries = [q.strip() for q in rephrased.rephrased_queries.split(';')]
        queries.append(query_or_queries)
        
        print(f"   - Expanded to Queries: {list(set(queries))}")

        # 2. Retrieve documents for all rephrased queries
        passages = []
        for query in set(queries): # Use set to avoid duplicate API calls
            if not query: continue
            query_vector = self.openai_client.embeddings.create(input=[query], model=self.embedding_model).data[0].embedding
            search_results = self.qdrant_client.search(collection_name=self.collection_name, query_vector=query_vector, limit=k)
            passages.extend(result.payload['text'] for result in search_results)
        
        # 3. Return the unique set of retrieved passages for the re-ranker
        return dspy.Prediction(passages=list(dict.fromkeys(passages)))

class GeminiReranker(dspy.Module):
    def __init__(self, api_key: str):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model_name = "models/embedding-001"

    def forward(self, query: str, passages: List[str], k: int = 5) -> List[str]:
        try:
            doc_embeddings = genai.embed_content(model=self.model_name, content=passages, task_type="RETRIEVAL_DOCUMENT")["embedding"]
            query_embedding = genai.embed_content(model=self.model_name, content=[query], task_type="RETRIEVAL_QUERY")["embedding"]
            scores = np.dot(np.array(query_embedding), np.array(doc_embeddings).T).flatten()
            return [p for p, s in sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)][:k]
        except Exception as e:
            print(f"Error during Gemini re-ranking: {e}")
            return passages[:k]

# --- Agent "Tools" ---
class SimpleRAG(dspy.Module):
    def __init__(self, retriever, reranker):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    def forward(self, question):
        context = self.retriever(question).passages
        reranked_context = self.reranker(query=question, passages=context, k=3)
        pred = self.generate_answer(context=reranked_context, question=question)
        return dspy.Prediction(answer=pred.answer, context=reranked_context)

class ComparativeRAG(dspy.Module):
    def __init__(self, retriever, reranker):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        self.generate_comparison = dspy.ChainOfThought("context, question -> comparison")
    def forward(self, question):
        context = self.retriever(question).passages
        reranked_context = self.reranker(query=question, passages=context, k=7)
        pred = self.generate_comparison(context=reranked_context, question=question)
        return dspy.Prediction(answer=pred.comparison, context=reranked_context)

class MultiStepRAG(dspy.Module):
    def __init__(self, simple_rag_agent):
        super().__init__()
        self.decomposer = dspy.ChainOfThought("complex_question -> sub_questions")
        self.synthesizer = dspy.ChainOfThought("original_question, qa_pairs -> final_answer")
        self.simple_rag_agent = simple_rag_agent
    def forward(self, question):
        sub_questions = self.decomposer(complex_question=question).sub_questions.split(';')
        qa_pairs = ""
        for sub_q in sub_questions:
            if sub_q.strip():
                print(f"   - Answering sub-question: '{sub_q.strip()}'")
                sub_answer = self.simple_rag_agent(question=sub_q.strip()).answer
                qa_pairs += f"Sub-Question: {sub_q.strip()}\nAnswer: {sub_answer}\n\n"
        print("   - Synthesizing final answer...")
        final_pred = self.synthesizer(original_question=question, qa_pairs=qa_pairs)
        return dspy.Prediction(answer=final_pred.final_answer, context=[qa_pairs])
        
# --- The Orchestrator Agent ---
class OrchestratorAgent(dspy.Module):
    def __init__(self, simple_agent, comparative_agent, multi_step_agent):
        super().__init__()
        self.classifier = dspy.Predict("question -> intent", n=1)
    def forward(self, question):
        prompt = f"Classify the user's question. Choices: Factual, Comparative, Multi-step. Question: {question}"
        intent_pred = self.classifier(question=prompt)
        user_intent = intent_pred.intent
        print(f"--- Detected Intent: '{user_intent}' ---")
        if "Comparative" in user_intent:
            return self.comparative_agent(question=question)
        elif "Multi-step" in user_intent:
            return self.multi_step_agent(question=question)
        else:
            return self.simple_agent(question=question)

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Advanced Agentic RAG API", version="5.0")

@app.on_event("startup")
def startup_event():
    print("--- Server is starting up... ---")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not all([openai_api_key, gemini_api_key, qdrant_url, qdrant_collection_name]):
        raise ValueError("One or more environment variables are missing.")

    llm = dspy.OpenAI(model='gpt-4o', api_key=openai_api_key, max_tokens=1500)
    dspy.settings.configure(lm=llm)
    print("DSPy's LLM configured.")
    
    # Use the new retriever with query expansion
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

# --- API DATA MODELS with Chat History ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []

class QueryResponse(BaseModel):
    answer: str
    context: list[str]

# NEW: A module to condense the history and new question
condense_question_with_history = dspy.ChainOfThought("chat_history, new_question -> standalone_question")

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Advanced Agentic RAG API!"}

@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    print(f"\nReceived new question: '{request.question}'")
    
    final_question = request.question
    # NEW: If there's chat history, condense the question
    if request.chat_history:
        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in request.chat_history])
        print("--- Condensing question with chat history ---")
        condensed = condense_question_with_history(chat_history=history_str, new_question=request.question)
        final_question = condensed.standalone_question
        print(f"--- Standalone question: '{final_question}' ---")

    # The rest of the pipeline uses the final, context-aware question
    orchestrator = app.state.orchestrator
    prediction = orchestrator(final_question)
    return QueryResponse(answer=prediction.answer, context=prediction.context)