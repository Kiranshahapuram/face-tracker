"""
Database connection and operations module.
Handles PostgreSQL CRUD, connection pooling, pgvector, and database recovery.
"""

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_values
import numpy as np
import json
import logging
from contextlib import contextmanager
# pgvector support
from pgvector.psycopg2 import register_vector

class Database:
    def __init__(self, config):
        self.config = config
        
        logging.info("Connecting to Database...")
        db_conf = self.config.database
        conn_kwargs = {
            "host": db_conf.host,
            "port": db_conf.port,
            "dbname": db_conf.dbname,
            "user": db_conf.user,
            "password": db_conf.password if db_conf.password else "postgres"
        }
            
        self.pool = ThreadedConnectionPool(
            db_conf.pool_min,
            db_conf.pool_max,
            **conn_kwargs
        )
        
        # Ensure schema exists on first connection
        self._init_schema()

        # Register pgvector
        with self.get_connection() as conn:
            register_vector(conn)

    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def _init_schema(self):
        with open("schema.sql", "r") as f:
            schema = f.read()
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema)
            conn.commit()

    def register_session(self, video_source: str, config_snapshot: dict) -> str:
        """Create a new session record. Return session_id UUID."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sessions (video_source, config_snapshot) VALUES (%s, %s) RETURNING id",
                    (video_source, json.dumps(config_snapshot))
                )
                session_id = str(cur.fetchone()[0])
            conn.commit()
        return session_id

    def end_session(self, session_id: str) -> None:
        """Set ended_at = now() on session."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE sessions SET ended_at = now() WHERE id = %s", (session_id,))
            conn.commit()

    def register_face(self, embedding: np.ndarray) -> str:
        """Insert new face with embedding. Return face_id UUID."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO faces (embedding) VALUES (%s) RETURNING id",
                    (embedding,)
                )
                face_id = str(cur.fetchone()[0])
            conn.commit()
        return face_id

    def update_embedding(self, face_id: str, new_embedding: np.ndarray, sample_count: int) -> None:
        """Update running-mean embedding and sample_count for existing face."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE faces 
                       SET embedding = %s, sample_count = %s, embedding_version = embedding_version + 1, last_seen = now(), visit_count = visit_count + 1
                       WHERE id = %s""",
                    (new_embedding, sample_count, face_id)
                )
            conn.commit()

    def log_event_pending(self, face_id: str, event_type: str, image_path: str,
                          track_id: int, similarity_score: float,
                          frame_number: int, session_id: str) -> int:
        """Insert event with write_status='pending'. Return event id."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO events 
                       (face_id, event_type, image_path, write_status, track_id, similarity_score, frame_number, session_id) 
                       VALUES (%s, %s, %s, 'pending', %s, %s, %s, %s) RETURNING id""",
                    (face_id, event_type, image_path, track_id, similarity_score, frame_number, session_id)
                )
                event_id = cur.fetchone()[0]
            conn.commit()
        return event_id

    def complete_event(self, event_id: int) -> None:
        """Set write_status='complete' on event."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE events SET write_status = 'complete' WHERE id = %s", (event_id,))
            conn.commit()

    def load_all_embeddings(self) -> dict:
        """Return {face_id: np.ndarray} for all registered faces. Called on startup."""
        embeddings = {}
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, embedding FROM faces")
                for row in cur.fetchall():
                    embeddings[str(row[0])] = np.array(row[1])
        return embeddings

    def get_pending_events(self) -> list:
        """Return all events where write_status='pending'. For startup recovery."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, image_path FROM events WHERE write_status = 'pending'")
                return cur.fetchall()

    def delete_event(self, event_id: int) -> None:
        """Hard delete an event. Used in startup recovery for unresolvable pending events."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM events WHERE id = %s", (event_id,))
            conn.commit()

    def increment_visitor_count(self, session_id: str) -> int:
        """Atomically increment unique_visitors on session. Return new count."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE sessions SET unique_visitors = unique_visitors + 1 WHERE id = %s RETURNING unique_visitors", (session_id,))
                cnt = cur.fetchone()[0]
            conn.commit()
        return cnt

    def get_unique_visitor_count(self, session_id: str) -> int:
        """Return current unique_visitors for session."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT unique_visitors FROM sessions WHERE id = %s", (session_id,))
                res = cur.fetchone()
                return res[0] if res else 0

    def find_similar_face(self, embedding: np.ndarray, threshold: float) -> tuple:
        """Query pgvector for nearest face using cosine similarity.
        Return (face_id, similarity_score) or (None, 0.0) if none above threshold."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # pgvector cosine op is <=>, which returns distance.
                # similarity = 1 - distance. So distance < (1 - threshold).
                max_distance = 1.0 - threshold
                
                # We want nearest neighbor order by embedding <=> %s
                cur.execute(
                    """SELECT id, 1 - (embedding <=> %s::vector) AS sim
                       FROM faces
                       WHERE (embedding <=> %s::vector) <= %s
                       ORDER BY embedding <=> %s::vector LIMIT 1""",
                    (embedding, embedding, max_distance, embedding)
                )
                res = cur.fetchone()
                if res:
                    return str(res[0]), float(res[1])
                return None, 0.0

    def run_startup_recovery(self) -> None:
        """Recovery routine."""
        import os
        pending = self.get_pending_events()
        logging.info(f"Running startup recovery. Found {len(pending)} pending events.")
        for event_id, image_path in pending:
            if os.path.exists(image_path):
                self.complete_event(event_id)
                logging.info(f"Recovered pending event {event_id}: marked complete.")
            else:
                self.delete_event(event_id)
                logging.info(f"Recovered pending event {event_id}: deleted (missing image).")
