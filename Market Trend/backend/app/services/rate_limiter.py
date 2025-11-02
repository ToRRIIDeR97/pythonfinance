import time

class TokenBucket:
    def __init__(self, rate: int, per_seconds: int):
        self.capacity = rate
        self.tokens = rate
        self.per_seconds = per_seconds
        self.last_refill = time.time()

    def consume(self, amount: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        refill = int(elapsed // self.per_seconds) * self.capacity
        if refill > 0:
            self.tokens = min(self.capacity, self.tokens + refill)
            self.last_refill = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False