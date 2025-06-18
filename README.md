# Advanced Agentic RAG System with DSPy

This project implements a sophisticated, multi-agent Retrieval-Augmented Generation (RAG) system built with the DSPy framework. It goes beyond simple Q&A to create a conversational agent that can reason about user intent, decompose complex problems, and use a multi-layered retrieval process to provide high-quality, context-aware answers from a document set.

The entire system is served via a professional, production-ready FastAPI application.

 
*(Note: You can create and link to a diagram of the workflow we discussed to make this even more impressive).*

---

## ✨ Key Features

This is not a standard RAG pipeline. It incorporates several advanced techniques to achieve superior performance and intelligence:

*   **Vision-Based Data Ingestion (Zerox):** Instead of naive text extraction, the indexing pipeline uses **Zerox** to perform vision-based OCR on PDFs. It interprets layouts, tables, and structures, generating clean Markdown for a high-quality knowledge base.
*   **Agentic Orchestrator:** The core of the system is an `OrchestratorAgent` that first classifies the user's intent (`Factual`, `Comparative`, `Multi-step`) before acting.
*   **Specialized Agent "Tools":** The orchestrator routes queries to a toolbox of specialized agents, each optimized for a specific task:
    *   `SimpleRAG`: For direct, factual questions.
    *   `ComparativeRAG`: For "compare and contrast" questions, using a specialized prompt.
    *   `MultiStepRAG`: For complex queries, it automatically decomposes the problem into sub-questions, answers each one, and synthesizes a final answer.
*   **Advanced Retrieval Pipeline:**
    *   **Query Expansion:** Uses an LLM to rephrase the user's query into multiple variations to cast a wider, more effective net for finding relevant documents.
    *   **Re-ranking:** A custom **Gemini-based re-ranker** sifts through the initial retrieved passages to select the top 5-7 most relevant ones, ensuring the final context is highly focused.
*   **Conversational Memory:** The API endpoint supports chat history, allowing the agent to understand and correctly answer ambiguous follow-up questions.
*   **Production-Ready API:** The entire system is served via a **FastAPI** application with robust error handling, usage logging, and a clean, installable package structure.

---

## 🛠️ Tech Stack

*   **Framework:** DSPy (`dspy-ai`)
*   **LLMs:** OpenAI (`gpt-4o` for generation), Google Gemini (`embedding-001` for re-ranking)
*   **Vector Database:** Qdrant (running in Docker)
*   **Data Ingestion:** Zerox (`py-zerox`) for vision-based OCR
*   **API Server:** FastAPI & Uvicorn
*   **Environment:** Conda & Python 3.10+

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

*   Docker and Docker Compose are installed and running.
*   Conda is installed.
*   You are in a WSL (Windows Subsystem for Linux) environment or a native Linux/macOS terminal.

### 2. Initial Setup

**Clone and set up the environment:**

```bash
# Clone the repository (if applicable)
# git clone ...
# cd agentic-rag

# Create and activate the Conda environment
conda create -n agentic-rag python=3.10 -y
conda activate agentic-rag

# Install all system and Python dependencies
sudo apt-get update && sudo apt-get install -y poppler-utils
pip install -r requirements.txt # (You would create this file)
# Or install manually:
# pip install dspy-ai openai qdrant-client python-dotenv google-generativeai numpy fastapi uvicorn "pydantic<2" py-zerox langchain

# Install the project in editable mode to make local modules importable
pip install -e .