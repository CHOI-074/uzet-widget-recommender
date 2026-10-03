from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.batch.precompute import build_user_states
from src.config import load_catalog
from src.model.als_trainer import build_interactions, to_confidence, train
from src.model.evaluator import leave_one_out_split
from src.store.candidate_store import FileCandidateStore


def sample_events():
    rows = []
    for i in range(4):
        for j, widget in enumerate(["acct_balance", "savings", "stock_status"]):
            rows.append(dict(user_id=f"u{i}", persona_id="rookie", widget_id=widget,
                             timestamp=datetime(2025, 10, 1) + timedelta(hours=j+i),
                             action="click"))
    return pd.DataFrame(rows)


def test_training_store_reload_and_api(tmp_path, monkeypatch):
    events = sample_events()
    interactions = build_interactions(events, load_catalog())
    assert interactions.matrix.shape == (4, len(load_catalog().ids))
    model = train(interactions, factors=4, iterations=3)
    states = build_user_states(events, model, top_k=25)
    path = tmp_path / "candidates.pkl"
    store = FileCandidateStore(path)
    store.bulk_put(states)
    store.close()
    monkeypatch.setattr(api, "STORE_PATH", path)
    monkeypatch.setattr(api, "BACKEND", "file")
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        for user, source in [("u0", "personalized"), ("new-user", "fallback")]:
            response = client.get(f"/recommend/{user}", params={"n": 6, "now": "2025-10-15T01:30:00Z"})
            assert response.status_code == 200
            data = response.json()
            assert data["source"] == source
            assert len(data["widgets"]) == 6
            assert "market_open" in data["fired_rules"]
            assert len({w["widget_id"] for w in data["widgets"]}) == 6
            if source == "fallback":
                assert "prior" in data["widgets"][0]["parts"]
            for w in data["widgets"]:
                assert w["score"] == pytest.approx(sum(w["parts"].values()), abs=1e-5)
        assert client.get("/recommend/u0?n=0").status_code == 422


def test_confidence_and_empty_training():
    assert to_confidence(np.array([1, 2]), 2, False).tolist() == [3, 5]
    assert to_confidence(np.array([1]), 2, True)[0] == pytest.approx(1 + 2*np.log(2))
    with pytest.raises(ValueError):
        to_confidence(np.array([-1]), 2, True)
    events = sample_events().iloc[:0]
    with pytest.raises(ValueError):
        train(build_interactions(events, load_catalog()))


def test_split_excludes_later_impressions_with_duplicate_indices():
    events = sample_events()
    future = events.iloc[[0]].copy()
    future["action"] = "impression"
    future["timestamp"] = datetime(2025, 11, 1)
    train_df, held = leave_one_out_split(pd.concat([events, future]))
    for row in held.itertuples():
        assert (train_df.loc[train_df.user_id == row.user_id, "timestamp"] < row.timestamp).all()
