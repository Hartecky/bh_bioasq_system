from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.snippets = []

    def build(self, questions):
        texts = []
        for q in questions:
            for snippet in q.snippets:
                self.snippets.append(snippet)

        for snippet in self.snippets:
            texts.append(snippet["text"])

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            )
        dim = embeddings.shape[1]  # Number of columns = length of a vector
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
