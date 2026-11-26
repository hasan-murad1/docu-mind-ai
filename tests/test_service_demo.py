"""Tests for demo loading and separate quota ids in the service."""

import pytest

import service as service_module
from fakes import make_service
from service import ServiceError
from usage_limiter import MAX_QUESTIONS_PER_DAY


@pytest.fixture
def env(tmp_path):
    return make_service(tmp_path / "uploads")


def test_quota_is_shared_by_sessions_with_the_same_quota_id(env):
    # Two browser tabs from the same IP share one daily quota
    for i in range(MAX_QUESTIONS_PER_DAY):
        env.service.ask(f"tab{i % 2}", "question", quota_id="203.0.113.5")
    with pytest.raises(ServiceError) as caught:
        env.service.ask("tab9", "question", quota_id="203.0.113.5")
    assert caught.value.status == 429


def test_documents_follow_the_session_not_the_quota_id(env):
    env.service.ask("tab1", "question", quota_id="203.0.113.5")
    assert env.sessions.visitors == ["tab1"]


def test_usage_reads_the_quota_id(env):
    env.service.ask("tab1", "question", quota_id="203.0.113.5")
    usage = env.service.usage("tab2", quota_id="203.0.113.5")
    assert usage["questions_left"] == MAX_QUESTIONS_PER_DAY - 1


def test_load_demo_documents_once(env, tmp_path):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "handbook.pdf").write_bytes(b"%PDF-1.4 demo")
    (demo_dir / "readme.txt").write_text("ignored")
    assert env.service.load_demo_documents(demo_dir) == 1
    assert env.stored_in == [("handbook.pdf", "demo_test")]
    assert env.service.load_demo_documents(demo_dir) == 0  # already loaded


def test_load_demo_documents_without_folder(env, tmp_path):
    assert env.service.load_demo_documents(tmp_path / "missing") == 0


def test_demo_handbook_is_in_the_repo():
    assert (service_module.DEMO_DOCS_DIR / "Velora_Travel_Handbook.pdf").is_file()