import pytest

from src.tests.fixtures.consts import TestConsts


@pytest.fixture
def mock_model_name_env_var(monkeypatch):
    monkeypatch.setenv("NER_MODEL_NAME", TestConsts.test_model_name)
