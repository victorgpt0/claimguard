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
        instance.chat_completion.return_value.choices = [
            MagicMock(message=MagicMock(content="  This is the narrative.  "))]

        result = generator.generate("Patient had fever.", _FakeVerdict("inconsistent"), "Definition X")

    assert result == "This is the narrative."
    prompt = instance.chat_completion.call_args.kwargs["messages"][0]["content"]
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


def test_vllm_provider_loads_model_once_and_reuses_it():
    import sys
    from types import SimpleNamespace

    fake_vllm = MagicMock()
    fake_vllm.LLM.return_value.chat.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text="  Narrative.  ")])]
    generator = AuditNarrativeGenerator(provider="vllm", model_name="fake-awq")
    with patch.dict(sys.modules, {"vllm": fake_vllm}):
        first = generator.generate("Notes A.", _FakeVerdict("inconsistent"), "Definition X")
        generator.generate("Notes B.", _FakeVerdict("inconsistent"), "Definition Y")

    assert first == "Narrative."
    assert fake_vllm.LLM.call_count == 1
    assert fake_vllm.LLM.call_args.kwargs["dtype"] == "half"
    prompt = fake_vllm.LLM.return_value.chat.call_args_list[1].args[0][0]["content"]
    assert "Notes B." in prompt and "Definition Y" in prompt
