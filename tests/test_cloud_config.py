"""Guards for the cloud deploy: secrets, dependencies, upload cap and provider setup."""

import tomllib
from pathlib import Path

import pytest
import requests

import rag_pipeline
from fakes import make_service
from rag_pipeline import ProviderNotConfigured
from service import ServiceError
from upload_security import MAX_UPLOAD_BYTES
from usage_limiter import MAX_QUESTIONS_PER_DAY

ROOT = Path(__file__).resolve().parents[1]


def test_ask_groq_without_a_key_fails_clearly(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "GROQ_API_KEY", None)

    def no_network(*args, **kwargs):
        raise AssertionError("No request should be sent without a key")

    monkeypatch.setattr(requests, "post", no_network)
    with pytest.raises(ProviderNotConfigured):
        rag_pipeline.ask_groq("hello")


def test_missing_key_becomes_a_friendly_503(tmp_path):
    env = make_service(tmp_path)
    env.agent.error = ProviderNotConfigured("no key")
    with pytest.raises(ServiceError) as caught:
        env.service.ask("alice", "question")
    assert caught.value.status == 503
    assert "set up" in caught.value.message
    assert env.questions.remaining_today("alice") == MAX_QUESTIONS_PER_DAY


def test_secrets_never_go_to_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored
    assert ".streamlit/secrets.toml" in ignored


def test_requirements_stay_light():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "torch" not in requirements
    assert "sentence-transformers" not in requirements


def test_streamlit_upload_cap_matches_the_service():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] * 1024 * 1024 == MAX_UPLOAD_BYTES


def test_demo_folder_has_no_answer_key():
    names = [p.name.lower() for p in (ROOT / "demo_docs").iterdir()]
    assert names and not any("question" in n for n in names)