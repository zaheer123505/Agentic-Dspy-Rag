#data_modules.py will hold our QdrantRetriever and GeminiReranker.
# src/components/data_modules.py
import os
import dspy
import google.generativeai as genai
from qdrant_client import QdrantClient
from openai import OpenAI
from typing import List, Optional
import numpy as np

# --- Reusable Custom Modules ---
class QdrantRetrieverWithExpansion(dspy.Retrieve):
    def __init__(self, collection_name: str, url: str, openai_api_key: str, k: int = 20):
        super().__init__(k=k)
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-large"
        # NEW: A module to rephrase the query for better retrieval
        self.query_rephraser = dspy.ChainOfThought("question -> rephrased_queries", n=1)

    def forward(self, query_or_queries, k: Optional[int] = None):
        k = k if k is not None else self.k
        
        # 1. Rephrase the original query into multiple versions
        rephrased = self.query_rephraser(question=f"Generate 3 diverse questions related to this query, keeping the original intent. Separate them with a semi-colon. Query: {query_or_queries}")
        queries = [q.strip() for q in rephrased.rephrased_queries.split(';')]
        queries.append(query_or_queries) # Add the original query to the list
        print(f"   - Expanded to Queries: {queries}")

        # 2. Retrieve documents for all rephrased queries
        passages = []
        for query in set(queries): # Use set to avoid duplicate queries
            query_vector = self.openai_client.embeddings.create(input=[query], model=self.embedding_model).data[0].embedding
            search_results = self.qdrant_client.search(collection_name=self.collection_name, query_vector=query_vector, limit=k)
            passages.extend(result.payload['text'] for result in search_results)
        
        # 3. Return the unique set of retrieved passages
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





