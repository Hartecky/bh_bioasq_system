from google import genai
from dotenv import load_dotenv
import os, time
from src.generation.base_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    def __init__(self, model="gemini-2.0-flash"):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 30  # czekaj 30 sekund przed retry
                    print(f"Rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise