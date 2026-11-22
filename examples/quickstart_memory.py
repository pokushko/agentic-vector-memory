"""
Quickstart Example for Agentic Vector Memory 🧠💾
Demonstrating episodic insertion, temporal decay retrieval, and compaction.
"""

import numpy as np
from vector_memory import EpisodicVectorStore

def main():
    store = EpisodicVectorStore(dimension=4, decay_lambda=0.001, alpha=0.7)

    print("=== 1. Indexing Agent Observations ===")
    store.add(
        memory_id="obs_001",
        text="Observed upstream API latency spike > 500ms during market open",
        embedding=[0.9, 0.1, 0.0, 0.2],
        metadata={"category": "telemetry", "severity": "warning"}
    )
    store.add(
        memory_id="obs_002",
        text="Successfully routed block order via VWAP dynamic order slicer",
        embedding=[0.1, 0.9, 0.2, 0.0],
        metadata={"category": "execution", "status": "filled"}
    )

    print(f"Total memories stored: {len(store.records)}")

    print("\n=== 2. Hybrid Querying (Semantic + Recency) ===")
    query_vector = [0.85, 0.15, 0.0, 0.1]
    results = store.query(query_vector=query_vector, top_k=2)

    for rank, (rec, score) in enumerate(results, 1):
        print(f"Rank {rank} (Score: {score:.4f}):")
        print(f"  ID: {rec.id}")
        print(f"  Text: {rec.text}")
        print(f"  Metadata: {rec.metadata}")

    print("\n=== 3. Memory Consolidation ===")
    evicted = store.consolidate(min_retention_score=0.1, max_age_seconds=86400)
    print(f"Consolidation complete. Evicted {evicted} stale records.")

if __name__ == "__main__":
    main()
