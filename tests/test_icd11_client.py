import pytest
import requests

from src.icd11.client import ICD11APIClient, ICD11APIError


class _FakeResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


def test_get_code_definition_happy_path(monkeypatch):
    client = ICD11APIClient(base_url="http://localhost:8081", release="2026-01")

    def fake_get(url, headers=None, timeout=None):
        if "codeinfo" in url:
            return _FakeResponse({"stemId": "http://localhost:8081/icd/entity/12345"})
        return _FakeResponse({
            "title": {"@value": "Some Condition"},
            "definition": {"@value": "A description of the condition."},
            "inclusion": [{"@value": "related term"}],
            "exclusion": [],
        })

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_code_definition("1A00")
    assert result.title == "Some Condition"
    assert result.definition == "A description of the condition."
    assert result.inclusion == ["related term"]


def test_get_code_definition_raises_after_retries(monkeypatch):
    client = ICD11APIClient(base_url="http://localhost:8081", max_retries=2, backoff_seconds=0)

    def failing_get(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", failing_get)
    with pytest.raises(ICD11APIError):
        client.get_code_definition("1A00")


def test_health_check_false_when_unreachable(monkeypatch):
    client = ICD11APIClient(base_url="http://localhost:8081", max_retries=1, backoff_seconds=0)
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    assert client.health_check() is False
