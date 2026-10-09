import pytest
from cli.chat import ConversationSession


def test_conversation_session_formatting():
    session = ConversationSession(system_prompt="Test system.")
    session.add_user_message("Hello!")
    session.add_assistant_message("Hi there!")
    session.add_user_message("How are you?")

    formatted = session.format_prompt()

    assert "System: Test system." in formatted
    assert "User: Hello!" in formatted
    assert "Assistant: Hi there!" in formatted
    assert "User: How are you?" in formatted
    assert formatted.endswith("Assistant: ")

    session.clear()
    assert len(session.history) == 0
