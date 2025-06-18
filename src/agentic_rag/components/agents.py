# src/agentic_rag/components/agents.py
# This file defines the "brains" of our application: the various DSPy agent modules.
# It includes the specialized "tools" and the main orchestrator that controls them.

import dspy
from agentic_rag.components.data_modules import QdrantRetrieverWithExpansion, GeminiReranker, log_api_usage

# --- Tool 1: The Simple RAG Agent ---
class SimpleRAG(dspy.Module):
    """Handles direct, factual questions using a standard retrieve-rerank-generate pipeline."""
    def __init__(self, retriever, reranker):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        # Defines the task for the LLM: given context and a question, produce an answer.
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 1. Retrieve a broad set of passages.
        context = self.retriever(question).passages
        # 2. Re-rank to get the most relevant passages.
        reranked_context = self.reranker(query=question, passages=context, k=3)
        # 3. Generate a final answer from the refined context.
        prediction = self.generate_answer(context=reranked_context, question=question)

        # Log the token usage for this LLM call.
        try:
            usage = dspy.settings.lm.history[-1]['response'].usage
            log_api_usage("OpenAI", dspy.settings.lm.kwargs.get('model', 'unknown'), usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        except Exception:
            print("[USAGE LOG] Could not parse usage data for SimpleRAG.")
            
        return dspy.Prediction(answer=prediction.answer, context=reranked_context)

# --- Tool 2: The Comparative RAG Agent ---
class ComparativeRAG(dspy.Module):
    """A specialized agent for handling "compare and contrast" questions."""
    def __init__(self, retriever, reranker):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        # Uses a specialized signature to guide the LLM towards a structured comparison.
        self.generate_comparison = dspy.ChainOfThought("context, question -> comparison")

    def forward(self, question):
        context = self.retriever(question).passages
        # We retrieve more context (k=7) for comparisons as they often require more information.
        reranked_context = self.reranker(query=question, passages=context, k=7)
        prediction = self.generate_comparison(context=reranked_context, question=question)
        
        # Log token usage.
        try:
            usage = dspy.settings.lm.history[-1]['response'].usage
            log_api_usage("OpenAI", dspy.settings.lm.kwargs.get('model', 'unknown'), usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        except Exception:
            print("[USAGE LOG] Could not parse usage data for ComparativeRAG.")

        # Map the 'comparison' output field to 'answer' for a consistent API response.
        return dspy.Prediction(answer=prediction.comparison, context=reranked_context)

# --- Tool 3: The Multi-Step RAG Agent ---
class MultiStepRAG(dspy.Module):
    """An advanced agent that handles complex questions by decomposing them into simpler sub-questions."""
    def __init__(self, simple_rag_agent):
        super().__init__()
        # Module to break down a complex question.
        self.decomposer = dspy.ChainOfThought("complex_question -> sub_questions")
        # Module to synthesize a final answer from a series of Q&A pairs.
        self.synthesizer = dspy.ChainOfThought("original_question, qa_pairs -> final_answer")
        # This agent uses another agent (SimpleRAG) as its tool to answer the sub-questions.
        self.simple_rag_agent = simple_rag_agent

    def forward(self, question):
        # 1. Decompose the original complex question.
        decomposed = self.decomposer(complex_question=question)
        sub_questions = [q.strip() for q in decomposed.sub_questions.split(';') if q.strip()]
        
        # 2. Loop through each sub-question and use the SimpleRAG tool to find its answer.
        qa_pairs = ""
        for sub_q in sub_questions:
            print(f"   - Answering sub-question: '{sub_q}'")
            prediction = self.simple_rag_agent(question=sub_q)
            qa_pairs += f"Sub-Question: {sub_q}\nAnswer: {prediction.answer}\n\n"
        
        # 3. Synthesize a final, comprehensive answer using all the collected information.
        print("   - Synthesizing final answer...")
        final_pred = self.synthesizer(original_question=question, qa_pairs=qa_pairs)
        
        # The 'context' for this agent is the full reasoning trace of sub-questions and answers.
        return dspy.Prediction(answer=final_pred.final_answer, context=[qa_pairs])
        
# --- The Main Orchestrator Agent ---
class OrchestratorAgent(dspy.Module):
    """
    The master agent that controls the workflow. It analyzes the user's query and
    routes it to the most appropriate specialized agent (tool).
    """
    def __init__(self, simple_agent, comparative_agent, multi_step_agent):
        super().__init__()
        # The classifier module determines the user's intent.
        self.classifier = dspy.Predict("question -> intent", n=1)
        # The toolbox of available agents.
        self.simple_rag_agent = simple_agent
        self.comparative_rag_agent = comparative_agent
        self.multi_step_rag_agent = multi_step_agent
    
    def forward(self, question):
        # 1. Classify the intent of the user's question.
        prompt = f"Classify the user's question. Choices: Factual, Comparative, Multi-step. Question: {question}"
        intent_prediction = self.classifier(question=prompt)
        user_intent = intent_prediction.intent
        print(f"--- Detected Intent: '{user_intent}' ---")

        # 2. Route the question to the correct agent based on the classified intent.
        if "Comparative" in user_intent:
            print("--- Routing to: ComparativeRAG Agent ---")
            prediction = self.comparative_rag_agent(question=question)
        elif "Multi-step" in user_intent:
            print("--- Routing to: MultiStepRAG Agent ---")
            prediction = self.multi_step_rag_agent(question=question)
        else: # Default to the Factual agent.
            print("--- Routing to: SimpleRAG Agent ---")
            prediction = self.simple_rag_agent(question=question)
        
        # 3. Attach the detected intent to the final prediction object for API response.
        prediction.intent = user_intent
        return prediction