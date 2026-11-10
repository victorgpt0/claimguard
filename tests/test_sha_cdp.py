"""The simulated SHA claims platform (#65): seeding without labels, SHA's processing times,
the clock, and the paid-claims API with its cursor and tokens."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sha_cdp import clock, db, seed
from sha_cdp.app import app

READ = {"Authorization": "Bearer read-key"}
ADMIN = {"Authorization": "Bearer admin-token"}


def claims_set(n=60):
    dates = pd.date_range("2025-10-01", periods=n, freq="2D")
    return pd.DataFrame({
        "claim_id": [f"LCLM{i:06d}" for i in range(1, n + 1)], "provider_id": "PRV0001", "facility_level": 4,
        "facility_type": "sub_county_hospital", "member_id": [f"LMBR{i:06d}" for i in range(1, n + 1)],
        "beneficiary_type": "principal", "patient_age": [30.0] * (n - 1) + [None], "patient_sex": "female",
        "icd11_code": "JB20.0", "claim_amount": 10000.0, "service_date": dates.strftime("%Y-%m-%d"),
        "discharge_date": [d.strftime("%Y-%m-%d") for d in dates[:-1] + timedelta(days=1)] + [None],
        "clinical_notes": "Spontaneous vaginal delivery of a live infant, mother stable, discharged home.",
        "is_fraud": 0, "fraud_typology": "none", "label": "consistent", "is_upcoded": False,
        "original_icd11_code": "JB20.0", "has_text_noise": False,
    })


@pytest.fixture
def platform(tmp_path, monkeypatch):
    monkeypatch.setenv("SHA_CDP_DATABASE_URL", f"sqlite:///{tmp_path / 'cdp.sqlite'}")
    monkeypatch.setenv("SHA_CDP_API_KEY", "read-key")
    monkeypatch.setenv("SHA_CDP_ADMIN_TOKEN", "admin-token")
    db.engine.cache_clear()
    eng = db.engine()
    seed.load(claims_set(), eng)
    yield eng, TestClient(app)
    db.engine.cache_clear()


def test_schedule_follows_sha_timelines_and_drops_labels():
    out = seed.schedule(claims_set(), seed=1)
    assert list(out.columns) == seed.KEEP + ["lodged_at", "approved_at", "paid_at"]
    assert not {"is_fraud", "fraud_typology", "label", "original_icd11_code"} & set(out.columns)
    discharged = pd.to_datetime(out["discharge_date"]).fillna(pd.to_datetime(out["service_date"]))
    assert ((out["lodged_at"].dt.normalize() - discharged).dt.days.between(0, 7)).all()      # reg. 58(1)
    assert ((out["paid_at"] - out["approved_at"]).dt.days <= 90).all()                         # 90-day rule
    assert (out["approved_at"] > out["lodged_at"]).all() and out["paid_at"].is_monotonic_increasing


def test_seed_refuses_to_load_twice_unless_replacing(platform):
    eng, _ = platform
    with pytest.raises(ValueError, match="already holds"):
        seed.load(claims_set(), eng)
    assert seed.load(claims_set(10), eng, replace=True) == 10


def test_api_needs_the_read_key(platform):
    _, client = platform
    assert client.get("/api/v1/claims/paid").status_code == 401
    assert client.get("/api/v1/claims/paid", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_clock_starts_before_the_first_payment_and_releases_claims(platform):
    eng, client = platform
    assert client.get("/api/v1/claims/paid", headers=READ).json()["claims"] == []
    client.post("/api/v1/clock", headers=ADMIN, json={"action": "advance", "days": 45})
    body = client.get("/api/v1/claims/paid", headers=READ, params={"limit": 500}).json()
    as_of = pd.Timestamp(body["as_of"])
    assert body["claims"] and all(pd.Timestamp(c["paid_at"]) <= as_of for c in body["claims"])
    assert "is_fraud" not in body["claims"][0] and "label" not in body["claims"][0]


def test_cursor_pages_without_repeats_or_gaps(platform):
    eng, client = platform
    clock.advance(eng, 400)
    seen, params = [], {"limit": 7}
    while True:
        body = client.get("/api/v1/claims/paid", headers=READ, params=params).json()
        seen += [c["claim_id"] for c in body["claims"]]
        if not body["has_more"]:
            break
        params = {"limit": 7, **body["next"]}
    assert len(seen) == len(set(seen)) == 60


def test_only_the_operator_moves_the_clock(platform):
    eng, client = platform
    before = clock.now(eng)
    assert client.post("/api/v1/clock", headers=READ, json={"action": "advance", "days": 1}).status_code == 401
    assert client.post("/api/v1/clock", headers=ADMIN, json={"action": "advance", "days": -1}).status_code == 422
    played = client.post("/api/v1/clock", headers=ADMIN, json={"action": "play", "days_per_minute": 30}).json()
    assert played["playing"] and played["days_per_minute"] == 30
    assert not client.post("/api/v1/clock", headers=ADMIN, json={"action": "pause"}).json()["playing"]
    assert clock.now(eng) >= before


def test_stats_count_what_has_happened_by_the_clock(platform):
    eng, client = platform
    clock.advance(eng, 60)
    stats = client.get("/api/v1/stats", headers=READ).json()
    assert stats["claims_held"] == 60 and stats["lodged"] >= stats["approved"] >= stats["paid"] > 0
    assert stats["awaiting_payment"] == stats["lodged"] - stats["paid"]


def test_clock_read_before_seeding_does_not_release_every_claim(tmp_path, monkeypatch):
    # On a fresh deployment the ingest worker polls the empty platform before anyone seeds it.
    # That read creates the clock at today's date; the first load must reset it, or every claim
    # is already "paid" and all audit periods close at once.
    monkeypatch.setenv("SHA_CDP_DATABASE_URL", f"sqlite:///{tmp_path / 'cdp.sqlite'}")
    db.engine.cache_clear()
    eng = db.engine()
    db.metadata.create_all(eng)
    assert clock.now(eng).year >= 2026          # empty platform: the clock starts at today
    seed.load(claims_set(), eng)
    with eng.connect() as conn:
        first_payment = conn.execute(db.claims.select().order_by(db.claims.c.paid_at)).first().paid_at
    assert clock.now(eng) < first_payment     # nothing paid yet: the periods close as the clock moves
    db.engine.cache_clear()
