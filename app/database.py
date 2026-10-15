import logging
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from app.config import get_settings

logger = logging.getLogger("face_service.db")


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(get_settings().db_file))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema and add the embedding column to databases from older versions."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                embedding BLOB
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "embedding" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN embedding BLOB")
        conn.commit()
    logger.info("Database initialized at %s", get_settings().db_file)


def add_user(name: str, image_path: str, embedding: np.ndarray | None) -> int:
    blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, image_path, embedding) VALUES (?, ?, ?)",
            (name, image_path, blob),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_users() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, name, image_path FROM users ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def count_users() -> int:
    with get_db() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def delete_user(user_id: int) -> str | None:
    """Delete a user and return the stored image path, or None if the id does not exist."""
    with get_db() as conn:
        row = conn.execute(
            "DELETE FROM users WHERE id = ? RETURNING image_path", (user_id,)
        ).fetchone()
        conn.commit()
    return row["image_path"] if row else None


def load_embeddings() -> tuple[list[dict], np.ndarray]:
    """Return (user metadata, N x D float32 matrix) for every user that has an embedding."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, image_path, embedding FROM users WHERE embedding IS NOT NULL"
        ).fetchall()
    users = [{"id": r["id"], "name": r["name"], "image_path": r["image_path"]} for r in rows]
    if not rows:
        return users, np.empty((0, 0), dtype=np.float32)
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return users, matrix


def users_without_embedding() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, image_path FROM users WHERE embedding IS NULL").fetchall()
    return [dict(row) for row in rows]


def set_embeddings(embeddings: dict[int, np.ndarray]) -> None:
    """Store several embeddings in one statement batch."""
    if not embeddings:
        return
    with get_db() as conn:
        conn.executemany(
            "UPDATE users SET embedding = ? WHERE id = ?",
            [(vec.astype(np.float32).tobytes(), uid) for uid, vec in embeddings.items()],
        )
        conn.commit()
