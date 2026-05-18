from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

GAMMA_BASE = "https://gamma-api.polymarket.com"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False


class GammaClient:
    def __init__(self, base_url: str = GAMMA_BASE, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GammaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def fetch_markets(self, **params: Any) -> list[dict]:
        r = self._client.get("/markets", params=params)
        r.raise_for_status()
        return r.json()

    def iter_markets(self, limit: int = 500, **filters: Any) -> Iterator[list[dict]]:
        offset = 0
        while True:
            page = self.fetch_markets(limit=limit, offset=offset, **filters)
            if not page:
                return
            yield page
            offset += limit
