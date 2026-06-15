import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Refresh the token slightly before it actually expires to avoid using a
# token that expires mid-request.
_EXPIRY_BUFFER_SECONDS = 60


class OAuth2TokenManager:
    """Thread-safe client-credentials OAuth2 token cache with automatic refresh."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self._client = client or httpx.Client(timeout=15.0)
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_token(self) -> Optional[str]:
        """Return a cached token, refreshing it first if it's missing or near expiry."""
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            return self._refresh()

    def refresh(self) -> Optional[str]:
        """Force a token refresh, bypassing the cache (used for health checks)."""
        with self._lock:
            return self._refresh()

    def _refresh(self) -> Optional[str]:
        try:
            response = self._client.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": self.scope,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("OAuth2 token refresh failed: %s", exc)
            self._token = None
            self._expires_at = 0.0
            return None

        payload = response.json()
        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 0)

        self._token = token
        self._expires_at = time.monotonic() + max(0, expires_in - _EXPIRY_BUFFER_SECONDS)
        return token
