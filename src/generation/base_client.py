class BaseLLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclass must implement generate()")