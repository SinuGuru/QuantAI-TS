import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

DB_PATH = Path("app.db")

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
        UNIQUE(user_id, name),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        tokens INTEGER DEFAULT 0,
        requests INTEGER DEFAULT 0,
        cost REAL DEFAULT 0.0,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, date)
    );""")
    conn.commit()
    return conn

def create_user(conn, username: str, password: str, role: str = "user") -> Dict[str, Any]:
    from auth import hash_password
    pw_hash = hash_password(password)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (username, pw_hash, role))
        conn.commit()
        return {"id": c.lastrowid, "username": username, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError("User exists")

def get_user_by_username(conn, username: str) -> Optional[Dict[str, Any]]:
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}

def save_conversation_db(conn, user_id: int, name: str, messages: List[Dict[str, Any]]) -> None:
    import json
    c = conn.cursor()
    payload = json.dumps(messages, ensure_ascii=False, indent=2)
    try:
        c.execute("""
        INSERT INTO conversations (user_id, name, data) VALUES (?, ?, ?)
        ON CONFLICT(user_id, name) DO UPDATE SET data=excluded.data, created_at=(DATETIME('now'))
        """, (user_id, name, payload))
        conn.commit()
    except sqlite3.OperationalError:
        c.execute("DELETE FROM conversations WHERE user_id = ? AND name = ?", (user_id, name))
        c.execute("INSERT INTO conversations (user_id, name, data) VALUES (?, ?, ?)", (user_id, name, payload))
        conn.commit()

def get_user_conversations(conn, user_id: int) -> List[Tuple[str, str]]:
    c = conn.cursor()
    c.execute("SELECT name, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    return c.fetchall()

def load_conversation_db(conn, user_id: int, name: str) -> Optional[List[Dict[str, Any]]]:
    import json
    c = conn.cursor()
    c.execute("SELECT data FROM conversations WHERE user_id = ? AND name = ? LIMIT 1", (user_id, name))
    row = c.fetchone()
    if not row:
        return None
    return json.loads(row[0])

def add_usage(conn, user_id: int, tokens: int = 0, requests: int = 1, cost: float = 0.0) -> None:
    from datetime import date
    today = date.today().isoformat()
    c = conn.cursor()
    try:
        c.execute("""
        INSERT INTO usage (user_id, date, tokens, requests, cost)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, date) DO UPDATE SET
          tokens = tokens + excluded.tokens,
          requests = requests + excluded.requests,
          cost = cost + excluded.cost
        """, (user_id, today, tokens, requests, cost))
    except sqlite3.OperationalError:
        c.execute("SELECT id FROM usage WHERE user_id = ? AND date = ?", (user_id, today))
        if c.fetchone():
            c.execute("UPDATE usage SET tokens = tokens + ?, requests = requests + ?, cost = cost + ? WHERE user_id = ? AND date = ?",
                      (tokens, requests, cost, user_id, today))
        else:
            c.execute("INSERT INTO usage (user_id, date, tokens, requests, cost) VALUES (?, ?, ?, ?, ?)",
                      (user_id, today, tokens, requests, cost))
    conn.commit()