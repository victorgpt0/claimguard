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
    # Response shapes captured from the live whoicd/icd-api container (release 2026-01):
    # stemId is a canonical id.who.int URI, and inclusion/exclusion items wrap their text
    # in a "label" object alongside reference URIs.
    client = ICD11APIClient(base_url="http://localhost:8081", release="2026-01")
    requested_urls = []

    def fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        if "codeinfo" in url:
            return _FakeResponse({"stemId": "http://id.who.int/icd/release/11/2026-01/mms/12345"})
        return _FakeResponse({
            "title": {"@language": "en", "@value": "Some Condition"},
            "definition": {"@language": "en", "@value": "A description of the condition."},
            "inclusion": [{"label": {"@language": "en", "@value": "related term"}}],
            "exclusion": [{
                "label": {"@language": "en", "@value": "Excluded condition"},
                "foundationReference": "http://id.who.int/icd/entity/999",
                "linearizationReference": "http://id.who.int/icd/release/11/2026-01/mms/999",
            }],
        })

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_code_definition("1A00")
    assert result.title == "Some Condition"
    assert result.definition == "A description of the condition."
    assert result.inclusion == ["related term"]
    assert result.exclusion == ["Excluded condition"]
    # The entity fetch must stay on the local container, never go to WHO's cloud API.
    assert requested_urls[1] == "http://localhost:8081/icd/release/11/2026-01/mms/12345"
    assert result.definition_source == ""


def test_residual_category_inherits_parent_definition(monkeypatch):
    # Live shape for CA40.Z "Pneumonia, organism unspecified": no definition, one parent.
    client = ICD11APIClient(base_url="http://localhost:8081", release="2026-01")
    entities = {
        "/icd/release/11/2026-01/mms/142052508/unspecified": {
            "title": {"@value": "Pneumonia, organism unspecified"},
            "parent": ["http://id.who.int/icd/release/11/2026-01/mms/142052508"],
        },
        "/icd/release/11/2026-01/mms/142052508": {
            "code": "CA40",
            "title": {"@value": "Pneumonia"},
            "definition": {"@value": "An inflammation of the lung parenchyma."},
        },
    }

    def fake_get(url, headers=None, timeout=None):
        if "codeinfo" in url:
            return _FakeResponse({"stemId": "http://id.who.int/icd/release/11/2026-01/mms/142052508/unspecified"})
        return _FakeResponse(entities[url.removeprefix("http://localhost:8081")])

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_code_definition("CA40.Z")
    assert result.title == "Pneumonia, organism unspecified"
    assert result.definition == "An inflammation of the lung parenchyma."
    assert result.definition_source == "CA40"


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


def test_definition_walk_stops_at_chapter(monkeypatch):
    # Live shape for BD1Z "Heart failure, unspecified": the nearest definition is the
    # chapter's, which describes the chapter, not the condition.
    client = ICD11APIClient(base_url="http://localhost:8081", release="2026-01")
    entities = {
        "/icd/release/11/2026-01/mms/1": {"title": {"@value": "Heart failure, unspecified"},
                                           "parent": ["http://id.who.int/icd/release/11/2026-01/mms/2"]},
        "/icd/release/11/2026-01/mms/2": {"classKind": "chapter", "code": "11",
                                           "definition": {"@value": "This chapter includes..."}},
    }

    def fake_get(url, headers=None, timeout=None):
        if "codeinfo" in url:
            return _FakeResponse({"stemId": "http://id.who.int/icd/release/11/2026-01/mms/1"})
        return _FakeResponse(entities[url.removeprefix("http://localhost:8081")])

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_code_definition("BD1Z")
    assert result.definition == ""
    assert result.definition_source == ""
