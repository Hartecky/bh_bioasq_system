from google import genai
from dotenv import load_dotenv
import os
from src.generation.base_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    def __init__(self, model="gemini-2.5-flash"):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text