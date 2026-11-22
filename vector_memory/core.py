import time
import math
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class MemoryRecord:
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: float
    last_accessed_at: float
    access_count: int = 1

    def compute_temporal_weight(self, current_time: float, decay_lambda: float = 0.0001) -> float:
        """Computes exponential temporal decay: exp(-lambda * delta_t)."""
        delta_t = max(0.0, current_time - self.last_accessed_at)
        return math.exp(-decay_lambda * delta_t)

class EpisodicVectorStore:
    """
    Episodic & Semantic Vector Store for Autonomous AI Agents.
    Combines dense cosine similarity with exponential recency decay,
    providing biologically inspired memory retention and compaction.
    """

    def __init__(self, dimension: int = 128, decay_lambda: float = 0.0001, alpha: float = 0.7, db_path: Optional[str] = None):
        self.dimension = dimension
        self.decay_lambda = decay_lambda
        self.alpha = alpha  # Weight for semantic similarity vs recency
        self.db_path = db_path
        self.records: Dict[str, MemoryRecord] = {}
        
        if self.db_path:
            self._init_sqlite()
            self._load_from_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding TEXT,
                    metadata TEXT,
                    created_at REAL,
                    last_accessed_at REAL,
                    access_count INTEGER
                )
            """)
        conn.close()

    def _load_from_sqlite(self):
        if not self.db_path or not Path(self.db_path).exists():
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, embedding, metadata, created_at, last_accessed_at, access_count FROM memories")
        for row in cursor.fetchall():
            rec = MemoryRecord(
                id=row[0],
                text=row[1],
                embedding=json.loads(row[2]),
                metadata=json.loads(row[3]),
                created_at=row[4],
                last_accessed_at=row[5],
                access_count=row[6]
            )
            self.records[rec.id] = rec
        conn.close()

    def add(self, memory_id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> MemoryRecord:
        """Normalizes vector and inserts memory record."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        norm_emb = arr.tolist()

        now = time.time()
        record = MemoryRecord(
            id=memory_id,
            text=text,
            embedding=norm_emb,
            metadata=metadata or {},
            created_at=now,
            last_accessed_at=now,
            access_count=1
        )
        self.records[memory_id] = record

        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, text, embedding, metadata, created_at, last_accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.text,
                    json.dumps(record.embedding),
                    json.dumps(record.metadata),
                    record.created_at,
                    record.last_accessed_at,
                    record.access_count
                ))
            conn.close()

        return record

    def query(self, query_vector: List[float], top_k: int = 5, min_score: float = 0.0, filter_metadata: Dict[str, Any] = None) -> List[Tuple[MemoryRecord, float]]:
        """
        Retrieves top-k relevant memories combining cosine similarity and recency decay.
        Score = alpha * CosineSimilarity + (1 - alpha) * TemporalWeight
        """
        if not self.records:
            return []

        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        now = time.time()
        scored_records = []

        for rec in self.records.values():
            # Apply metadata filters
            if filter_metadata:
                match = all(rec.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            # Dense cosine similarity (dot product of unit vectors)
            v_arr = np.array(rec.embedding, dtype=np.float32)
            cosine_sim = float(np.dot(q_arr, v_arr))
            cosine_sim = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))  # Normalize to [0, 1]

            # Exponential recency decay
            recency = rec.compute_temporal_weight(now, self.decay_lambda)

            # Combined hybrid score
            hybrid_score = self.alpha * cosine_sim + (1.0 - self.alpha) * recency

            if hybrid_score >= min_score:
                scored_records.append((rec, hybrid_score))

        # Sort descending by hybrid score
        scored_records.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_records[:top_k]

        # Update access count & last_accessed_at for retrieved items
        for rec, _ in top_results:
            rec.access_count += 1
            rec.last_accessed_at = now

        return top_results

    def consolidate(self, min_retention_score: float = 0.2, max_age_seconds: float = 86400 * 30) -> int:
        """
        Memory compaction: Purges stale, forgotten, low-relevance episodic memories.
        Returns the count of evicted records.
        """
        now = time.time()
        evicted_ids = []

        for rec_id, rec in list(self.records.items()):
            age = now - rec.created_at
            recency = rec.compute_temporal_weight(now, self.decay_lambda)
            # Retain if accessed frequently or high recency
            importance = recency * (1.0 + math.log1p(rec.access_count))
            if age > max_age_seconds and importance < min_retention_score:
                evicted_ids.append(rec_id)

        for rec_id in evicted_ids:
            del self.records[rec_id]

        if self.db_path and evicted_ids:
            conn = sqlite3.connect(self.db_path)
            with conn:
                conn.executemany("DELETE FROM memories WHERE id = ?", [(rid,) for rid in evicted_ids])
            conn.close()

        return len(evicted_ids)
