import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

    def search(self, query, k = 5):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(query_embedding, k)
        return [self.snippets[i] for i in indices[0]]
    
    def save(self, directory):
       Path(directory).mkdir(parents=True, exist_ok=True)
       faiss.write_index(self.index, str(Path(directory) / "index.faiss"))
       with open(Path(directory) / "snippets.pkl", "wb") as f:
           pickle.dump(self.snippets, f)

    def load(self, directory):
        self.index = faiss.read_index(str(Path(directory) / "index.faiss"))

        with open(Path(directory) / "snippets.pkl", "rb") as f:
            self.snippets = pickle.load(f)


