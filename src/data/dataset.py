from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json

class Question(BaseModel):
    id: str
    type: str
    body: str
    snippets: list[dict]
    ideal_answer: Optional[list[str]] = None
    exact_answer: Optional[list | str] = None


class BioASQDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.questions = []
        self._load()

    def _load(self):
        with open(self.path) as f:
            data = json.load(f)

        for record in data["questions"]:
            self.questions.append(Question(**record))