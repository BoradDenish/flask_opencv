import sqlite3

import numpy as np

from app import database
from app.config import get_settings


def vec(*values):
    return np.array(values, dtype=np.float32)


def test_init_db_is_idempotent():
    database.init_db()
    database.init_db()
    assert database.count_users() == 0


def test_init_db_migrates_legacy_schema(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, image_path TEXT NOT NULL)")
    conn.execute("INSERT INTO users (name, image_path) VALUES ('old', 'static/old.jpg')")
    conn.commit()
    conn.close()
    monkeypatch.setenv("DB_FILE", str(legacy))

    database.init_db()

    assert [u["name"] for u in database.list_users()] == ["old"]
    assert [u["image_path"] for u in database.users_without_embedding()] == ["static/old.jpg"]


def test_add_list_and_delete_user():
    first = database.add_user("a", "static/a.jpg", vec(1, 0))
    second = database.add_user("b", "static/b.jpg", None)

    assert [u["id"] for u in database.list_users()] == [second, first]
    assert database.delete_user(first) == "static/a.jpg"
    assert database.delete_user(first) is None
    assert database.count_users() == 1


def test_load_embeddings_skips_users_without_one():
    database.add_user("a", "static/a.jpg", vec(1, 0))
    database.add_user("b", "static/b.jpg", None)
    database.add_user("c", "static/c.jpg", vec(0, 1))

    users, matrix = database.load_embeddings()

    assert [u["name"] for u in users] == ["a", "c"]
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32


def test_load_embeddings_empty():
    users, matrix = database.load_embeddings()
    assert users == [] and matrix.size == 0


def test_set_embeddings_updates_many_rows_at_once():
    ids = [database.add_user(n, f"static/{n}.jpg", None) for n in "abc"]

    database.set_embeddings({ids[0]: vec(1, 0), ids[2]: vec(0, 1)})

    assert [u["image_path"] for u in database.users_without_embedding()] == ["static/b.jpg"]


def test_sql_values_are_bound_not_formatted():
    nasty = "x'); DROP TABLE users;--"
    database.add_user(nasty, "static/x.jpg", None)
    assert database.list_users()[0]["name"] == nasty


def test_settings_follow_environment():
    assert get_settings().db_file.name == "test_users.db"
