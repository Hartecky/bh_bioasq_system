from pydantic import BaseModel
from typing import Optional

class Question(BaseModel):
    id: str
    type: str
    body: str
    snippets: list[dict]
    ideal_answer: Optional[list[str]]
    exact_answer: Optional[list]



