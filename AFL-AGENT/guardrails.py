import re
import threading
import time
from collections import defaultdict, deque

MAX_INPUT_CHARS = 500          # longest user message accepted
MAX_OUTPUT_CHARS = 1200        # longest answer body returned
MAX_TOOL_CALLS_PER_TURN = 2    # tool calls the model may make for one message
REQUESTS_PER_MINUTE = 8        # chat messages per session per minute
TOOL_CALLS_PER_10_MIN = 10     # ML tool executions per session per 10 minutes


class GuardrailError(Exception):
    """A safe, user-facing rejection message."""


class RateLimiter:
    """Sliding-window limiter keyed by session/thread id."""

    def __init__(self, max_calls: int, window_s: float):
        self.max_calls, self.window_s = max_calls, window_s
        self._calls = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str):
        """Return (allowed, seconds_until_next_slot)."""
        now = time.monotonic()
        with self._lock:
            q = self._calls[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.max_calls:
                return False, self.window_s - (now - q[0])
            q.append(now)
            return True, 0.0


REQUEST_LIMITER = RateLimiter(REQUESTS_PER_MINUTE, 60)
TOOL_LIMITER = RateLimiter(TOOL_CALLS_PER_10_MIN, 600)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET = re.compile(r"\b(gsk_|sk-|AIza|hf_)[A-Za-z0-9_\-]{10,}")


def sanitize_input(text: str) -> str:
    text = _CONTROL.sub("", text or "").strip()
    if not text:
        raise GuardrailError("Please type a question about AFL.")
    if len(text) > MAX_INPUT_CHARS:
        raise GuardrailError(f"That message is too long ({len(text)} characters). "
                             f"Please keep it under {MAX_INPUT_CHARS}.")
    return text


def clip_output(text: str) -> str:
    """Redact anything that looks like an API key, then cap the length at a sentence/line end."""
    text = _SECRET.sub("[redacted]", text or "")
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    cut = text[:MAX_OUTPUT_CHARS]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    if end > MAX_OUTPUT_CHARS * 0.6:
        cut = cut[:end + 1]
    return cut.rstrip() + " …(answer shortened)"


def wait_message(wait: float) -> str:
    return f"You're sending messages too quickly. Please wait about {int(wait) + 1} seconds and try again."