import json
import threading

from src.core.budget_guard import BudgetGuard


def _guard(tmp_path, limit=300):
    return BudgetGuard(limit=limit, path=tmp_path / "claude_quota.json")


def test_check_and_increment_within_limit(tmp_path):
    guard = _guard(tmp_path, limit=2)

    assert guard.check_and_increment() is True
    assert guard.check_and_increment() is True
    assert guard.check_and_increment() is False
    assert guard.get_remaining() == 0


def test_release_restores_quota(tmp_path):
    guard = _guard(tmp_path, limit=1)

    assert guard.check_and_increment() is True
    assert guard.get_remaining() == 0

    guard.release()
    assert guard.get_remaining() == 1
    assert guard.check_and_increment() is True


def test_count_persisted_to_file(tmp_path):
    path = tmp_path / "claude_quota.json"
    guard = BudgetGuard(limit=10, path=path)

    guard.check_and_increment()
    guard.check_and_increment()

    assert json.loads(path.read_text())["count"] == 2

    reloaded = BudgetGuard(limit=10, path=path)
    assert reloaded.get_remaining() == 8


def test_concurrent_calls_do_not_exceed_quota(tmp_path):
    guard = _guard(tmp_path, limit=1)

    results: list[bool] = []

    def _attempt():
        results.append(guard.check_and_increment())

    threads = [threading.Thread(target=_attempt) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert results.count(False) == 9
    assert guard.get_remaining() == 0
