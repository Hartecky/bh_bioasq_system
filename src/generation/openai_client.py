from openai import OpenAI
from dotenv import load_dotenv
import os, time
from src.generation.base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, model="gpt-4o"):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                    )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 30  # czekaj 30 sekund przed retry
                    print(f"Rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise                