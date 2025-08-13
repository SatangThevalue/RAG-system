import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

CHATLOG_URL = os.getenv("CHATLOG_POSTGRES_URL", "")
ENABLE = os.getenv("ENABLE_CHATLOG_POSTGRES", "false").lower() == "true"

_engine = None
_SessionLocal = None

def init_db():
    global _engine, _SessionLocal
    if not ENABLE or not CHATLOG_URL: return False
    _engine = create_engine(CHATLOG_URL, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    ddl = """
    CREATE TABLE IF NOT EXISTS chat_messages (
      id BIGSERIAL PRIMARY KEY,
      session_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      ts BIGINT NOT NULL
    );
    """
    with _engine.begin() as conn:
        conn.execute(text(ddl))
    return True

def save_message(session_id: str, role: str, content: str, ts: int):
    if not ENABLE or _engine is None: return
    with _engine.begin() as conn:
        conn.execute(
            text("INSERT INTO chat_messages(session_id, role, content, ts) VALUES (:s,:r,:c,:t)"),
            {"s": session_id, "r": role, "c": content, "t": ts},
        )