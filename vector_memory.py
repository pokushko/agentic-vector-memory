
import numpy as np
from typing import List, Tuple

class SimpleVectorStore:
    """
    A lightweight vector database implementation for Agent Memory.
    Uses cosine similarity for semantic retrieval.
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []

    def add(self, vector: List[float], text: str):
        self.vectors.append(np.array(vector) / np.linalg.norm(vector))
        self.metadata.append(text)

    def query(self, query_vector: List[float], top_k: int = 1) -> List[Tuple[float, str]]:
        if not self.vectors: return []
        
        q_norm = np.array(query_vector) / np.linalg.norm(query_vector)
        similarities = [np.dot(v, q_norm) for v in self.vectors]
        
        results = sorted(zip(similarities, self.metadata), reverse=True)[:top_k]
        return results

if __name__ == "__main__":
    vdb = SimpleVectorStore()
    # Mock embeddings for "Agentic" concepts
    vdb.add([1.0, 0.2, 0.1] + [0]*125, "Agent autonomy theory")
    vdb.add([0.1, 0.9, 0.3] + [0]*125, "Molecular dynamics simulation")
    
    query = [0.9, 0.1, 0.2] + [0]*125
    print("Query results for 'Agent-like' vector:")
    for score, text in vdb.query(query):
        print(f"[{score:.4f}] {text}")
