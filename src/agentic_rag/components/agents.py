# src/agentic_rag/components/agents.py
# UPGRADED WITH USAGE LOGGING

import dspy
from agentic_rag.components.data_modules import QdrantRetrieverWithExpansion, GeminiReranker

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
        prediction = self.generate_answer(context=reranked_context, question=question)

        # NEW: Log usage from the last LLM call
        try:
            usage = dspy.settings.lm.history[-1]['response'].usage
            log_api_usage("OpenAI", dspy.settings.lm.kwargs['model'], usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        except (IndexError, AttributeError, KeyError):
            print("[USAGE LOG] Could not parse usage data for SimpleRAG.")
            
        return dspy.Prediction(answer=pred.answer, context=reranked_context)

# ... (The ComparativeRAG and MultiStepRAG classes will be almost identical) ...
class ComparativeRAG(dspy.Module):
    def __init__(self, retriever, reranker):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        self.generate_comparison = dspy.ChainOfThought("context, question -> comparison")
    def forward(self, question):
        context = self.retriever(question).passages
        reranked_context = self.reranker(query=question, passages=context, k=7)
        prediction = self.generate_comparison(context=reranked_context, question=question)
        
        try:
            usage = dspy.settings.lm.history[-1]['response'].usage
            log_api_usage("OpenAI", dspy.settings.lm.kwargs['model'], usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        except (IndexError, AttributeError, KeyError):
            print("[USAGE LOG] Could not parse usage data for ComparativeRAG.")

        return dspy.Prediction(answer=prediction.comparison, context=reranked_context)

class MultiStepRAG(dspy.Module):
    def __init__(self, simple_rag_agent):
        super().__init__()
        # Note: We don't log usage for the decomposer/synthesizer here for simplicity,
        # but the sub-agent calls within will be logged.
        self.decomposer = dspy.ChainOfThought("complex_question -> sub_questions")
        self.synthesizer = dspy.ChainOfThought("original_question, qa_pairs -> final_answer")
        self.simple_rag_agent = simple_rag_agent
    def forward(self, question):
        decomposed = self.decomposer(complex_question=question).sub_questions
        sub_questions = [q.strip() for q in decomposed.split(';') if q.strip()]
        qa_pairs = ""
        for sub_q in sub_questions:
            print(f"   - Answering sub-question: '{sub_q}'")
            # The SimpleRAG agent will log its own usage internally
            sub_answer = self.simple_rag_agent(question=sub_q).answer
            qa_pairs += f"Sub-Question: {sub_q}\nAnswer: {sub_answer}\n\n"
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
        else: # Default to Factual
            return self.simple_agent(question=question)