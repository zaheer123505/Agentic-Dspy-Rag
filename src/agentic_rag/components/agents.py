# src/agentic_rag/components/agents.py
import dspy
# The path is now relative to the 'agentic_rag' package
from agentic_rag.components.data_modules import QdrantRetrieverWithExpansion, GeminiReranker

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

# ... (ComparativeRAG and MultiStepRAG can be defined here, same as before) ...
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
        else: # Default to Factual
            return self.simple_agent(question=question)
