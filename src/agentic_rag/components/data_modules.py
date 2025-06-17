# src/agentic_rag/components/data_modules.py
# UPGRADED WITH ERROR HANDLING AND USAGE LOGGING

import os
import dspy
import google.generativeai as genai
from qdrant_client import QdrantClient
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from typing import List, Optional
import numpy as np
import datetime



# --- NEW: A centralized logging function ---
def log_api_usage(provider: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
    """Prints a structured log of API token usage."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"[USAGE LOG] | {timestamp} | Provider: {provider} | Model: {model} | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total Tokens: {total_tokens}")

# --- UPGRADED: Retriever with error handling and logging ---
class QdrantRetrieverWithExpansion(dspy.Retrieve):
    def __init__(self, collection_name: str, url: str, openai_api_key: str, k: int = 20):
        super().__init__(k=k)
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-large"
        self.query_rephraser = dspy.ChainOfThought("question -> rephrased_queries", n=1)

    def forward(self, query_or_queries, k: Optional[int] = None):
        k = k if k is not None else self.k
        
        try:
            rephrase_prompt = f"Generate 3 diverse but related questions to the original query. Keep the core intent. Separate them with a semi-colon. Original Query: {query_or_queries}"
            rephrased = self.query_rephraser(question=rephrase_prompt)
            queries = [q.strip() for q in rephrased.rephrased_queries.split(';')]
            queries.append(query_or_queries)
        except OpenAIRateLimitError as e:
            # Catch OpenAI quota errors during rephrasing
            raise APIQuotaError("OpenAI quota exceeded during query rephrasing.") from e

        print(f"   - Retriever is using expanded queries: {list(set(queries))}")

        passages = []
        for query in set(queries):
            if not query: continue
            try:
                response = self.openai_client.embeddings.create(input=[query], model=self.embedding_model)
                usage = response.usage
                log_api_usage("OpenAI", self.embedding_model, usage.prompt_tokens, 0, usage.total_tokens)
                query_vector = response.data[0].embedding
            except OpenAIRateLimitError as e:
                # Catch OpenAI quota errors during embedding
                raise APIQuotaError("OpenAI quota exceeded during query embedding.") from e
            
            search_results = self.qdrant_client.search(collection_name=self.collection_name, query_vector=query_vector, limit=k)
            passages.extend(result.payload['text'] for result in search_results)
        
        return dspy.Prediction(passages=list(dict.fromkeys(passages)))

# --- UPGRADED: Reranker with error handling ---
class GeminiReranker(dspy.Module):
    def __init__(self, api_key: str):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model_name = "models/embedding-001"

    def forward(self, query: str, passages: List[str], k: int = 5) -> List[str]:
        try:
            # For Gemini, usage is not returned per call, but we can catch the error
            response = genai.embed_content(model=self.model_name, content=passages, task_type="RETRIEVAL_DOCUMENT")
            doc_embeddings = response["embedding"]
            response = genai.embed_content(model=self.model_name, content=[query], task_type="RETRIEVAL_QUERY")
            query_embedding = response["embedding"]

            scores = np.dot(np.array(query_embedding), np.array(doc_embeddings).T).flatten()
            return [p for p, s in sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)][:k]
        except Exception as e:
            # A broad catch for any Gemini API error, including quota issues
            if "quota" in str(e).lower():
                raise APIQuotaError("Gemini API quota exceeded during re-ranking.") from e
            print(f"An unknown error occurred during Gemini re-ranking: {e}")
            return passages[:k]