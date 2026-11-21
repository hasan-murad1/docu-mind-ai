"""Tests for per-visitor sessions: isolation, demo copy, expiry and cleanup."""

import uuid

import pytest

import agent
import vector_store
from sessions import SESSION_TTL_SECONDS, SessionStore
from vector_store import add_chunks_to_store, list_sources, search_similar_chunks


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup():
    """Unique prefix and demo collection per test; everything is deleted afterwards."""
    prefix = f"t{uuid.uuid4().hex[:8]}_"
    demo = f"demo_{uuid.uuid4().hex[:8]}"
    clock = FakeClock()

    def make(**kwargs):
        return SessionStore(clock=clock, prefix=prefix, demo_collection=demo, **kwargs)

    yield make, clock, demo
    for name in vector_store.list_collection_names():
        if name.startswith(prefix) or name == demo:
            vector_store.delete_collection(name)


def test_each_visitor_gets_their_own_collection(setup):
    make, _, _ = setup
    store = make()
    a, b = store.collection_for("alice"), store.collection_for("bob")
    assert a != b
    assert store.collection_for("alice") == a


def test_uploads_are_isolated(setup):
    make, _, _ = setup
    store = make()
    a, b = store.collection_for("alice"), store.collection_for("bob")
    add_chunks_to_store(["Alice's secret plan is to launch in May."], "alice.pdf", collection_name=a)
    assert list_sources(b) == []
    assert search_similar_chunks("secret plan", top_k=3, collection_name=b) == []
    assert list_sources(a) == ["alice.pdf"]


def test_new_session_starts_with_demo_documents(setup):
    make, _, demo = setup
    add_chunks_to_store(["Velora Travel is a demo company."], "demo.pdf", collection_name=demo)
    store = make()
    mine = store.collection_for("alice")
    assert list_sources(mine) == ["demo.pdf"]
    add_chunks_to_store(["My own notes."], "mine.pdf", collection_name=mine)
    assert list_sources(demo) == ["demo.pdf"]  # the shared demo is never changed


def test_session_expires_after_two_hours(setup):
    make, clock, _ = setup
    store = make()
    first = store.collection_for("alice")
    clock.now += SESSION_TTL_SECONDS
    second = store.collection_for("alice")
    assert second != first
    assert first not in vector_store.list_collection_names()


def test_cleanup_works_after_a_restart(setup):
    make, clock, _ = setup
    old_name = make().collection_for("alice")
    clock.now += SESSION_TTL_SECONDS
    restarted = make()  # a new store knows nothing about alice
    assert restarted.cleanup_expired(force=True) == 1
    assert old_name not in vector_store.list_collection_names()


def test_max_sessions_drops_the_oldest(setup):
    make, clock, _ = setup
    store = make(max_sessions=2)
    oldest = store.collection_for("a")
    clock.now += 1
    store.collection_for("b")
    clock.now += 1
    store.collection_for("c")
    assert oldest not in vector_store.list_collection_names()
    assert len(store._session_names()) == 2


def test_end_session_deletes_files_now(setup):
    make, _, _ = setup
    store = make()
    name = store.collection_for("alice")
    assert store.end_session("alice") is True
    assert name not in vector_store.list_collection_names()
    assert store.end_session("alice") is False


def test_expires_in(setup):
    make, clock, _ = setup
    store = make()
    assert store.expires_in("alice") is None
    store.collection_for("alice")
    clock.now += 600
    assert store.expires_in("alice") == SESSION_TTL_SECONDS - 600


@pytest.mark.parametrize("bad_id", ["", None])
def test_rejects_bad_visitor_ids(setup, bad_id):
    make, _, _ = setup
    with pytest.raises(ValueError):
        make().collection_for(bad_id)


@pytest.mark.parametrize(
    "kwargs",
    [{"prefix": "BAD-"}, {"ttl_seconds": 0}, {"max_sessions": 0}],
)
def test_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        SessionStore(**kwargs)


def test_default_tools_search_the_given_collection(monkeypatch):
    seen = []
    monkeypatch.setattr(vector_store, "search_similar_chunks",
                        lambda q, top_k, collection_name: seen.append(collection_name) or [])
    monkeypatch.setattr(vector_store, "keyword_search",
                        lambda q, top_k, collection_name: seen.append(collection_name) or [])
    monkeypatch.setattr(vector_store, "metadata_search",
                        lambda q, source, top_k, collection_name: seen.append(collection_name) or [])
    for tool in agent.default_tools("visitor_x").values():
        tool("question", 3, "a.pdf")
    assert seen == ["visitor_x"] * 3