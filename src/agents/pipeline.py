from src.retrieval.retriever import Retriever
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import PromptBuilder


class QAPipeline:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()

    def answer(self, question):
        # Get snippets from retriever
        snippets = self.retriever.search(question.body, k=5)
        
        # 2. Build context
        context = "\n\n".join(s["text"] for s in snippets)
        
        # 3. Build prompt
        prompt = self.prompt_builder.build(
            question=question.body,
            context=context,
            question_type=question.type,
        )
        
        # 4. Generate answer through LLM
        return self.llm_client.generate(prompt)