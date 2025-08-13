import json, os, time
from typing import List, Dict
import redis as redis_lib

USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TTL = int(os.getenv("REDIS_TTL_SECONDS", "604800"))

class InMemoryStore:
    def __init__(self): self._data: Dict[str, List[Dict]] = {}
    def append(self, session_id: str, role: str, content: str):
        self._data.setdefault(session_id, []).append({"role": role, "content": content, "ts": int(time.time())})
    def history(self, session_id: str, limit: int = 10) -> List[Dict]:
        return self._data.get(session_id, [])[-limit:]
    def clear(self, session_id: str): self._data.pop(session_id, None)

class RedisStore:
    def __init__(self):
        self.r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    def _key(self, session_id: str) -> str: return f"chat:{session_id}:messages"
    def append(self, session_id: str, role: str, content: str):
        k = self._key(session_id)
        item = json.dumps({"role": role, "content": content, "ts": int(time.time())})
        self.r.rpush(k, item); 
        if REDIS_TTL > 0: self.r.expire(k, REDIS_TTL)
    def history(self, session_id: str, limit: int = 10) -> List[Dict]:
        k = self._key(session_id); n = self.r.llen(k)
        if n == 0: return []
        start = max(0, n - limit)
        raw = self.r.lrange(k, start, n)
        return [json.loads(x) for x in raw]
    def clear(self, session_id: str): self.r.delete(self._key(session_id))

def get_store():
    return RedisStore() if USE_REDIS else InMemoryStore()

def history_as_text(messages: List[Dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)