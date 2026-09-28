import pytest

from src.icd11.cache import ICD11Cache
from src.icd11.client import ICD11Definition


class _FakeClient:
    def __init__(self, fail_codes=None):
        self.fail_codes = fail_codes or set()
        self.calls = []

    def get_code_definition(self, code):
        self.calls.append(code)
        if code in self.fail_codes:
            from src.icd11.client import ICD11APIError
            raise ICD11APIError(f"simulated failure for {code}")
        return ICD11Definition(code=code, title=f"Title {code}", definition=f"Def {code}",
                                inclusion=[], exclusion=[])


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "icd11_cache.json"


def test_get_raises_keyerror_when_missing_and_no_client(cache_path):
    cache = ICD11Cache(cache_path)
    with pytest.raises(KeyError):
        cache.get("1A00")


def test_get_fetches_and_caches_on_miss(cache_path):
    cache = ICD11Cache(cache_path)
    client = _FakeClient()
    result = cache.get("1A00", client=client)
    assert result.title == "Title 1A00"
    assert client.calls == ["1A00"]

    # second call is a cache hit — no further client call
    result2 = cache.get("1A00", client=client)
    assert result2.title == "Title 1A00"
    assert client.calls == ["1A00"]


def test_save_and_load_round_trip(cache_path):
    cache = ICD11Cache(cache_path)
    cache.get("1A00", client=_FakeClient())

    reloaded = ICD11Cache(cache_path)
    assert "1A00" in reloaded
    assert reloaded.get("1A00").title == "Title 1A00"


def test_bulk_prefetch_skips_already_cached_and_reports_errors(cache_path):
    cache = ICD11Cache(cache_path)
    cache.get("1A00", client=_FakeClient())  # pre-populate one

    client = _FakeClient(fail_codes={"2B02"})
    errors = cache.bulk_prefetch(["1A00", "1A01", "2B02"], client)

    assert client.calls == ["1A01", "2B02"]  # 1A00 skipped, already cached
    assert set(errors.keys()) == {"2B02"}
    assert len(cache) == 2  # 1A00 + 1A01; 2B02 failed
