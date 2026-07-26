"""
vector_store.py — Portable Vector Store (SQLite + NumPy Vector Search & PostgreSQL Fallback)

Features:
  1. Default embedded SQLite vector database (vector_store.db) requiring zero external server setup.
  2. NumPy matrix-vector dot product cosine similarity for sub-millisecond search performance.
  3. Optional PostgreSQL + pgvector support enabled via environment variable USE_POSTGRES=1.
"""

import os
import json
import sqlite3
import importlib
import numpy as np


class VectorStore:
    """
    Portable Vector Database for storing and searching document text and embeddings.
    Supports in-memory cached matrix operations for high-speed similarity search.
    """
    def __init__(self, embedding_dim: int = 1024, db_path: str = "vector_store.db"):
        self.embedding_dim = embedding_dim
        self.db_path = db_path
        self.use_postgres = os.getenv("USE_POSTGRES", "0") in ("1", "true", "TRUE")

        # In-memory matrix cache for ultra-fast SQLite search
        self._ids: list[int] = []
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None  # shape: (N, dim), L2-normalized
        self._cache_valid = False

        if self.use_postgres:
            try:
                psycopg2_module = importlib.import_module("psycopg2")
                pool_module = importlib.import_module("psycopg2.pool")
            except ImportError:
                raise ImportError(
                    "To enable PostgreSQL mode (USE_POSTGRES=1), please run: "
                    "'pip install psycopg2-binary pgvector'"
                )

            self.db_params = {
                "dbname": os.getenv("PGDATABASE", "rag_db"),
                "user": os.getenv("PGUSER", "admin"),
                "password": os.getenv("PGPASSWORD", "gizlisifre"),
                "host": os.getenv("PGHOST", "localhost"),
                "port": os.getenv("PGPORT", "5432"),
            }
            self._pool = pool_module.SimpleConnectionPool(1, 10, **self.db_params)
            self._init_db_postgres()
        else:
            self._init_db_sqlite()

    # --- SQLITE IMPLEMENTATION ---

    def _get_sqlite_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db_sqlite(self):
        with self._get_sqlite_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS belgeler (
                    id INTEGER PRIMARY KEY,
                    metin TEXT NOT NULL,
                    vektor TEXT NOT NULL
                )
            """)
            conn.commit()

    def _ensure_cache(self):
        """Loads and normalizes all vectors into an in-memory NumPy matrix for fast searching."""
        if self._cache_valid and self._matrix is not None:
            return

        with self._get_sqlite_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, metin, vektor FROM belgeler ORDER BY id ASC")
            rows = cur.fetchall()

        if not rows:
            self._ids = []
            self._texts = []
            self._matrix = None
            self._cache_valid = True
            return

        self._ids = [r["id"] for r in rows]
        self._texts = [r["metin"] for r in rows]
        raw_vecs = [json.loads(r["vektor"]) for r in rows]
        matrix = np.array(raw_vecs, dtype=np.float32)

        # Pre-normalize matrix for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self._matrix = matrix / norms
        self._cache_valid = True

    # --- POSTGRES IMPLEMENTATION ---

    def _get_pg_conn(self):
        pgvector_mod = importlib.import_module("pgvector.psycopg2")
        conn = self._pool.getconn()
        pgvector_mod.register_vector(conn)
        return conn

    def _put_pg_conn(self, conn):
        self._pool.putconn(conn)

    def _init_db_postgres(self):
        conn = self._get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id BIGINT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR({self.embedding_dim})
                    );
                """)
            conn.commit()
        finally:
            self._put_pg_conn(conn)

    # --- PUBLIC API ---

    def reset(self):
        """Resets and clears all document tables before fresh ingestion."""
        if self.use_postgres:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DROP TABLE IF EXISTS documents;")
                conn.commit()
            finally:
                self._put_pg_conn(conn)
            self._init_db_postgres()
        else:
            with self._get_sqlite_conn() as conn:
                conn.execute("DROP TABLE IF EXISTS belgeler;")
                conn.commit()
            self._init_db_sqlite()
            self._cache_valid = False

    def add(self, doc_id: int, content: str, embedding: list[float]):
        """Inserts or updates a single document and embedding vector."""
        if self.use_postgres:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO documents (id, content, embedding) VALUES (%s, %s, %s) "
                        "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding",
                        (doc_id, content, np.array(embedding)),
                    )
                conn.commit()
            finally:
                self._put_pg_conn(conn)
        else:
            with self._get_sqlite_conn() as conn:
                conn.execute(
                    "INSERT INTO belgeler (id, metin, vektor) VALUES (?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET metin=excluded.metin, vektor=excluded.vektor",
                    (doc_id, content, json.dumps(embedding)),
                )
                conn.commit()
            self._cache_valid = False

    def add_batch(self, contents: list[str], embeddings: list[list[float]]):
        """Inserts multiple document records in batch."""
        if self.use_postgres:
            rows = [(i, c, np.array(e)) for i, (c, e) in enumerate(zip(contents, embeddings))]
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO documents (id, content, embedding) VALUES (%s, %s, %s)",
                        rows,
                    )
                conn.commit()
            finally:
                self._put_pg_conn(conn)
        else:
            rows = [(i, c, json.dumps(e)) for i, (c, e) in enumerate(zip(contents, embeddings))]
            with self._get_sqlite_conn() as conn:
                conn.executemany("INSERT INTO belgeler (id, metin, vektor) VALUES (?, ?, ?)", rows)
                conn.commit()
            self._cache_valid = False

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Executes dense vector similarity search and returns top-K matching documents.
        """
        if self.use_postgres:
            query = """
                SELECT id, content, 1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (np.array(query_embedding), np.array(query_embedding), top_k))
                    rows = cur.fetchall()
            finally:
                self._put_pg_conn(conn)
            return [{"id": row[0], "metin": row[1], "skor": float(row[2])} for row in rows]

        # SQLite implementation using accelerated NumPy matrix operations
        self._ensure_cache()
        if self._matrix is None or len(self._ids) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        scores = np.dot(self._matrix, q_vec)
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "id": self._ids[idx],
                "metin": self._texts[idx],
                "skor": float(scores[idx]),
            })
        return results

    def count(self) -> int:
        """Returns total document count in the vector database."""
        if self.use_postgres:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents;")
                    return cur.fetchone()[0]
            finally:
                self._put_pg_conn(conn)
        else:
            with self._get_sqlite_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM belgeler;")
                return cur.fetchone()[0]

    def close(self):
        """Closes any active database connection pools."""
        if self.use_postgres and hasattr(self, "_pool"):
            self._pool.closeall()