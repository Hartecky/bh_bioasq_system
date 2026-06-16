import requests
from src.generation.base_client import BaseLLMClient

class OllamaClient(BaseLLMClient):
    def __init__(self, model="phi3:mini", host="http://localhost:11434"):
        self.host = host
        self.model = model

    def generate(self, prompt):
        payload = {"model": self.model,
                   "prompt": prompt,
                   "stream": False}
        
        response = requests.post(url=f"{self.host}/api/generate", 
                                 json=payload)
        
        return response.json()["response"]