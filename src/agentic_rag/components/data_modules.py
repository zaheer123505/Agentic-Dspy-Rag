# src/agentic_rag/components/data_modules.py
# This file contains the modules responsible for data retrieval and refinement.
# These classes act as the bridge between our agentic logic and the external
# data sources like Qdrant and various embedding/re-ranking APIs.

import os
import dspy
import google.generativeai as genai
from qdrant_client import QdrantClient
from openai import OpenAI
from typing import List, Optional
import numpy as np
import datetime

# --- Centralized Logging Function ---
def log_api_usage(provider: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
    """
    A standardized function for logging API token usage.
    This helps in monitoring costs and understanding the application's workload.
    It can be easily extended later to log to a file or a monitoring service.
    """
    timestamp = datetime.datetime.now().isoformat()
    print(f"[USAGE LOG] | {timestamp} | Provider: {provider} | Model: {model} | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total Tokens: {total_tokens}")

# --- Custom Retriever Module with Query Expansion ---
class QdrantRetrieverWithExpansion(dspy.Retrieve):
    """
    An advanced retriever that enhances search quality through query expansion.
    Instead of just searching for the user's literal query, it first brainstorms
    several related queries to cast a wider net, improving the chances of
    finding the most relevant context (improving recall).
    """
    def __init__(self, collection_name: str, url: str, openai_api_key: str, k: int = 20):
        super().__init__(k=k)
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-large"
        # A lightweight DSPy module to generate query variations.
        self.query_rephraser = dspy.ChainOfThought("question -> rephrased_queries", n=1)

    def forward(self, query_or_queries, k: Optional[int] = None):
        k = k if k is not None else self.k
        
        # 1. Expand the original query into multiple, diverse variations.
        rephrase_prompt = f"Generate 3 diverse but related questions to the original query. Keep the core intent. Separate them with a semi-colon. Original Query: {query_or_queries}"
        rephrased = self.query_rephraser(question=rephrase_prompt)
        queries = [q.strip() for q in rephrased.rephrased_queries.split(';')]
        queries.append(query_or_queries) # Always include the original query.
        print(f"   - Retriever is using expanded queries: {list(set(queries))}")
        
        # 2. Retrieve documents for each generated query.
        passages = []
        for query in set(queries): # Use set() to avoid redundant searches for identical queries.
            if not query: continue
            
            # Embed the query using the same model as our indexing pipeline.
            response = self.openai_client.embeddings.create(input=[query], model=self.embedding_model)
            if response.usage:
                log_api_usage("OpenAI", self.embedding_model, response.usage.prompt_tokens, 0, response.usage.total_tokens)
            query_vector = response.data[0].embedding
            
            # Search Qdrant for the top k passages for this specific query.
            search_results = self.qdrant_client.search(collection_name=self.collection_name, query_vector=query_vector, limit=k)
            passages.extend(result.payload['text'] for result in search_results)
            
        # 3. Return a de-duplicated list of all found passages.
        # This becomes the candidate pool for the re-ranker.
        return dspy.Prediction(passages=list(dict.fromkeys(passages)))

# --- Custom Re-ranker Module ---
class GeminiReranker(dspy.Module):
    """
    A custom re-ranker module that uses the Google Gemini API.
    Its job is to take a large list of candidate passages and score each one
    for its specific relevance to the user's original query, improving precision.
    """
    def __init__(self, api_key: str):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model_name = "models/embedding-001"

    def forward(self, query: str, passages: List[str], k: int = 5) -> List[str]:
        try:
            # 1. Embed all candidate passages at once for efficiency.
            doc_embeddings = genai.embed_content(model=self.model_name, content=passages, task_type="RETRIEVAL_DOCUMENT")["embedding"]
            # 2. Embed the user's query.
            query_embedding = genai.embed_content(model=self.model_name, content=[query], task_type="RETRIEVAL_QUERY")["embedding"]
            
            # 3. Calculate the dot product (cosine similarity) to get relevance scores.
            scores = np.dot(np.array(query_embedding), np.array(doc_embeddings).T).flatten()
            
            # 4. Sort the original passages by their new scores and return the top 'k'.
            return [p for p, s in sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)][:k]
        except Exception as e:
            # Fallback in case the re-ranking API fails.
            print(f"Error during Gemini re-ranking: {e}")
            return passages[:k]