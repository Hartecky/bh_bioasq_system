from src.retrieval.retriever import Retriever
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import PromptBuilder
from src.agents.router import QuestionRouter

class QAPipeline:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()
        self.router = QuestionRouter()

    def answer(self, question, use_router: bool = False):
        snippets = self.retriever.search(question.body, k=5)
        context = "\n\n".join(s["text"] for s in snippets)
        
        # Użyj routera dla tekstu z zewnątrz, typu z datasetu dla ewaluacji
        q_type = self.router.classify(question.body) if use_router else question.type
        
        prompt = self.prompt_builder.build(
            question=question.body,
            context=context,
            question_type=q_type,
        )
        return self.llm_client.generate(prompt)