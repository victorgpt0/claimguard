from unittest.mock import MagicMock, patch

import pytest

from src.narrative.narrative_generator import AuditNarrativeGenerator


class _FakeVerdict:
    def __init__(self, verdict, confidence=0.9):
        self.verdict = verdict
        self.confidence = confidence


def test_generate_raises_for_consistent_verdict():
    generator = AuditNarrativeGenerator(provider="huggingface", model_name="fake-model")
    with pytest.raises(ValueError):
        generator.generate("notes", _FakeVerdict("consistent"), "definition")


def test_generate_huggingface_provider_builds_prompt_and_calls_client():
    generator = AuditNarrativeGenerator(provider="huggingface", model_name="fake-model")

    with patch("huggingface_hub.InferenceClient") as MockClient:
        instance = MockClient.return_value
        instance.text_generation.return_value = "  This is the narrative.  "

        result = generator.generate("Patient had fever.", _FakeVerdict("inconsistent"), "Definition X")

    assert result == "This is the narrative."
    call_args = instance.text_generation.call_args
    prompt = call_args.args[0]
    assert "Definition X" in prompt
    assert "Patient had fever." in prompt


def test_generate_openai_provider_calls_chat_completions():
    generator = AuditNarrativeGenerator(provider="openai", model_name="gpt-fake")

    with patch("openai.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        fake_message = MagicMock()
        fake_message.content = "  OpenAI narrative.  "
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        instance.chat.completions.create.return_value = MagicMock(choices=[fake_choice])

        result = generator.generate("notes", _FakeVerdict("inconsistent"), "definition")

    assert result == "OpenAI narrative."


def test_generate_unknown_provider_raises():
    generator = AuditNarrativeGenerator(provider="carrier-pigeon", model_name="x")
    with pytest.raises(ValueError):
        generator.generate("notes", _FakeVerdict("inconsistent"), "definition")
