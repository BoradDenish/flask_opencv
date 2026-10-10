import sqlite3
import pytest
import main


def test_init_db_creates_table(isolated_db):
    """Verify that init_db creates the users table with correct schema."""
    with main.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        table = cursor.fetchone()
        assert table is not None
        assert table["name"] == "users"

        cursor.execute("PRAGMA table_info(users)")
        columns = {col["name"]: col["type"] for col in cursor.fetchall()}
        assert "id" in columns
        assert "name" in columns
        assert "image_path" in columns


def test_init_db_is_idempotent(isolated_db):
    """Verify calling init_db multiple times does not drop or corrupt data."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Test User", "static/test.jpg"))
        conn.commit()

    # Re-run init_db
    main.init_db()

    with main.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM users")
        count = cursor.fetchone()["count"]
        assert count == 1


def test_get_db_row_factory(isolated_db):
    """Verify that get_db returns an sqlite3 connection with sqlite3.Row row_factory."""
    with main.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 'Alice' AS name, 42 AS age")
        row = cursor.fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["name"] == "Alice"
        assert row["age"] == 42


def test_user_crud_operations(isolated_db):
    """Verify full insert, select, and delete lifecycle for users in database."""
    # Create
    with main.get_db() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Hero", "static/hero.jpg"))
        user_id = cur.lastrowid
        conn.commit()
    assert user_id > 0

    # Read
    with main.get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cur.fetchone()
        assert user is not None
        assert user["name"] == "Hero"
        assert user["image_path"] == "static/hero.jpg"

    # Delete
    with main.get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

    # Verify deleted
    with main.get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        assert cur.fetchone() is None


def test_multiple_users_ordering(isolated_db):
    """Verify multiple users insertion and ordering by id desc."""
    users_data = [
        ("User 1", "static/u1.jpg"),
        ("User 2", "static/u2.jpg"),
        ("User 3", "static/u3.jpg")
    ]
    with main.get_db() as conn:
        for name, path in users_data:
            conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", (name, path))
        conn.commit()

    with main.get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM users ORDER BY id DESC")
        names = [row["name"] for row in cur.fetchall()]
        assert names == ["User 3", "User 2", "User 1"]
