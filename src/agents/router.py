# src/agents/router.py
import re


class QuestionRouter:
    """
    Classifies question type based on regex patterns.
    Returns: "yesno", "factoid", "list", or "summary"
    """

    YESNO_PATTERNS = [
        r"^(is|are|was|were|do|does|did|has|have|can|could|will|would|should)\s",
        r"^(does|do)\s",
    ]

    LIST_PATTERNS = [
        r"^(list|enumerate|name|which are|what are the)\s",
        r"\b(types of|kinds of|examples of|side effects of|causes of)\b",
    ]

    FACTOID_PATTERNS = [
        r"^(what is|what was|who is|who was|where is|when was|how many|how much)\s",
        r"^(what|who|where|when|how)\s",
    ]

    def classify(self, question: str) -> str:
        """Classifies a question string into a BioASQ question type."""
        q = question.lower().strip()

        for pattern in self.YESNO_PATTERNS:
            if re.search(pattern, q):
                return "yesno"

        for pattern in self.LIST_PATTERNS:
            if re.search(pattern, q):
                return "list"

        for pattern in self.FACTOID_PATTERNS:
            if re.search(pattern, q):
                return "factoid"

        return "summary"

    def classify_batch(self, questions: list[str]) -> list[str]:
        """Classifies a list of question strings."""
        return [self.classify(q) for q in questions]