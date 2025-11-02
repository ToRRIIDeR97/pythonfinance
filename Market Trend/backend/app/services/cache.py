import time
from typing import Any, Optional

try:
    from redis import Redis
except Exception:
    Redis = None

class InMemoryCache:
    def __init__(self):
        self.store = {}

    def set(self, key: str, value: Any, ttl: int = 60):
        self.store[key] = (value, time.time() + ttl)

    def get(self, key: str) -> Optional[Any]:
        item = self.store.get(key)
        if not item:
            return None
        value, expires = item
        if time.time() > expires:
            self.store.pop(key, None)
            return None
        return value


class CacheService:
    def __init__(self, redis_url: str):
        self._redis = None
        if Redis:
            try:
                self._redis = Redis.from_url(redis_url)
                # test connection
                self._redis.ping()
            except Exception:
                self._redis = None
        self._memory = InMemoryCache()

    def set(self, key: str, value: Any, ttl: int = 60):
        if self._redis:
            try:
                self._redis.setex(key, ttl, repr(value))
                return
            except Exception:
                pass
        self._memory.set(key, value, ttl)

    def get(self, key: str) -> Optional[Any]:
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    return eval(raw)  # value was repr()'d; safe for dev only
            except Exception:
                pass
        return self._memory.get(key)