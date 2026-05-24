from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
import random 

class Question(BaseModel):
    id: str
    type: str
    body: str
    snippets: list[dict]
    ideal_answer: Optional[list[str]] = None
    exact_answer: Optional[list | str] = None


class BioASQDataset:

    VALID_TYPES = {"yesno", "factoid", "list", "summary"}

    def __init__(self, path):
        self.path = Path(path)
        self.questions = []
        self._load()

    def _load(self):
        with open(self.path) as f:
            data = json.load(f)

        for record in data["questions"]:
            self.questions.append(Question(**record))

    def filter_by_type(self, question_type):
        if question_type not in self.VALID_TYPES:
            raise ValueError(f"Unknown type: '{question_type}'. Avalaible: {self.VALID_TYPES}")
    
        filtered_questions = []
        for q in self.questions:
            if q.type == question_type:
                filtered_questions.append(q)

        return filtered_questions
    
    def get_sample(self, n, question_type = None):
        if question_type:
            pool = self.filter_by_type(question_type)
        else:
            pool = self.questions

        return random.sample(pool, n)
