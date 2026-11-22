import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_memory.core import EpisodicVectorStore

def test_add_and_exact_vector_retrieval():
    store = EpisodicVectorStore(dimension=4, alpha=1.0)
    emb1 = [1.0, 0.0, 0.0, 0.0]
    emb2 = [0.0, 1.0, 0.0, 0.0]

    store.add("mem_1", "Crucial safety rule for chemical reactors", emb1, {"domain": "chemistry"})
    store.add("mem_2", "Agent task scheduling priority rules", emb2, {"domain": "orchestration"})

    results = store.query(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    best_record, score = results[0]
    assert best_record.id == "mem_1"
    assert score > 0.95

def test_metadata_filtering():
    store = EpisodicVectorStore(dimension=4, alpha=1.0)
    store.add("mem_1", "Result A", [1.0, 0.0, 0.0, 0.0], {"type": "fact"})
    store.add("mem_2", "Result B", [1.0, 0.1, 0.0, 0.0], {"type": "hypothesis"})

    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=5, filter_metadata={"type": "hypothesis"})
    assert len(results) == 1
    assert results[0][0].id == "mem_2"

def test_temporal_decay_favors_recent_memory():
    # Store with equal alpha (0.5 semantic, 0.5 recency)
    store = EpisodicVectorStore(dimension=4, decay_lambda=0.1, alpha=0.5)

    store.add("old_mem", "Old observation", [1.0, 0.0, 0.0, 0.0])
    # Manually warp old memory's timestamp
    store.records["old_mem"].last_accessed_at = time.time() - 100.0

    store.add("new_mem", "Recent observation", [0.8, 0.2, 0.0, 0.0])

    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
    # Even though old_mem has slightly higher cosine similarity, new_mem has far higher recency
    assert results[0][0].id == "new_mem"

def test_sqlite_persistence(tmp_path):
    db_file = str(tmp_path / "test_memory.db")
    store1 = EpisodicVectorStore(dimension=4, db_path=db_file)
    store1.add("perm_1", "Persisted agent insight", [0.5, 0.5, 0.5, 0.5], {"author": "Roman"})

    # Reload store from disk
    store2 = EpisodicVectorStore(dimension=4, db_path=db_file)
    assert "perm_1" in store2.records
    assert store2.records["perm_1"].text == "Persisted agent insight"
    assert store2.records["perm_1"].metadata["author"] == "Roman"

def test_consolidation_evicts_forgotten_memories():
    store = EpisodicVectorStore(dimension=4, decay_lambda=1.0)
    store.add("forgotten", "Transient noise", [0.1, 0.2, 0.3, 0.4])
    store.records["forgotten"].created_at = time.time() - 100000.0
    store.records["forgotten"].last_accessed_at = time.time() - 100000.0
    store.records["forgotten"].access_count = 1

    evicted = store.consolidate(min_retention_score=0.5, max_age_seconds=100.0)
    assert evicted == 1
    assert "forgotten" not in store.records
