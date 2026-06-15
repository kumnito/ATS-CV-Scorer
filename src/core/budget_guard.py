import json
import os
import threading
from pathlib import Path

# Budget global (process-wide) d'appels Claude — protège la clé API limitée à
# $10 sur la démo HF Spaces. Partagé par tous les usages billables (feedback
# texte ET Vision LLM, cascade niveau 3), quelle que soit l'entrée point
# (UI Gradio ou API REST), pour qu'aucun chemin ne contourne le quota.
CLAUDE_CALLS_LIMIT = 300

# Compteur persisté hors-process (survit aux rechargements de l'app dans le
# même conteneur HF Spaces) et protégé par un verrou contre les accès
# concurrents (UI Gradio + API FastAPI multithreads).
QUOTA_FILE_PATH = Path(os.environ.get("CLAUDE_QUOTA_FILE", "/tmp/claude_quota.json"))


class BudgetGuard:
    """Garde-fou centralisé pour le quota global d'appels Claude facturables.

    Thread-safe et persisté sur disque, partagé par l'UI (app.py) et l'API
    REST (server.py) afin qu'aucun point d'entrée ne puisse contourner le
    budget de la clé Anthropic.
    """

    def __init__(self, limit: int = CLAUDE_CALLS_LIMIT, path: Path = QUOTA_FILE_PATH) -> None:
        self._limit = limit
        self._path = path
        self._lock = threading.Lock()
        self._count = self._load()

    def _load(self) -> int:
        try:
            return int(json.loads(self._path.read_text()).get("count", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps({"count": self._count}))
        except OSError:
            pass

    def check_and_increment(self) -> bool:
        """Atomically check the quota and reserve one call slot."""
        with self._lock:
            if self._count >= self._limit:
                return False
            self._count += 1
            self._save()
            return True

    def release(self) -> None:
        """Roll back a reservation when the underlying API call failed."""
        with self._lock:
            self._count = max(0, self._count - 1)
            self._save()

    def get_remaining(self) -> int:
        with self._lock:
            return max(0, self._limit - self._count)

    @property
    def limit(self) -> int:
        return self._limit


budget_guard = BudgetGuard()
