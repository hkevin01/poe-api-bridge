import pytest
from server import count_tokens, count_message_tokens
import fastapi_poe as fp


def test_count_tokens_empty():
    """Test that empty string returns 0 tokens"""
    assert count_tokens("") == 0


def test_count_tokens_simple():
    """Test counting tokens for simple strings"""
    # Most English words are 1-2 tokens
    assert count_tokens("hello") == 1
    assert count_tokens("hello world") == 2

    # Test longer text
    text = (
        "This is a longer piece of text that should have more than just a few tokens."
    )
    assert count_tokens(text) > 10

    # Test that whitespace is counted
    assert count_tokens("hello   world") > count_tokens("hello world")


def test_count_tokens_special_characters():
    """Test counting tokens with special characters"""
    # Special characters and emojis often have their own token counts
    assert count_tokens("hello!@#$%^&*()") > count_tokens("hello")

    # Emojis often take multiple tokens
    assert count_tokens("hello 😊") > count_tokens("hello")


def test_count_tokens_different_languages():
    """Test counting tokens in different languages"""
    # Non-English languages typically have higher token counts
    english = "hello world"
    chinese = "你好世界"  # "Hello world" in Chinese

    # Chinese characters typically take more tokens in the tokenizer
    assert count_tokens(chinese) > count_tokens(english)


def test_count_tokens_code():
    """Test counting tokens in code snippets"""
    code = """def hello_world():
    print("Hello, world!")
    return True
"""
    # Code should have multiple tokens
    assert count_tokens(code) > 10


def test_count_message_tokens_empty():
    """Test that empty message list returns 3 tokens (overhead)"""
    messages = []
    result = count_message_tokens(messages)

    # Should only have the overhead tokens
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 3


def test_count_message_tokens_basic():
    """Test counting tokens in basic message list"""
    messages = [
        fp.ProtocolMessage(role="user", content="Hello, how are you?"),
        fp.ProtocolMessage(role="bot", content="I'm doing well, thank you!"),
    ]

    result = count_message_tokens(messages)

    # Verify structure
    assert "prompt_tokens" in result
    assert "completion_tokens" in result
    assert "total_tokens" in result

    # User message + overhead should be counted in prompt_tokens
    assert result["prompt_tokens"] > 3

    # Bot message should be counted in completion_tokens
    assert result["completion_tokens"] > 0

    # Total should be sum of prompt and completion
    assert (
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    )


def test_count_message_tokens_complex():
    """Test counting tokens in more complex message exchange"""
    messages = [
        fp.ProtocolMessage(role="system", content="You are a helpful AI assistant."),
        fp.ProtocolMessage(role="user", content="Tell me about token counting."),
        fp.ProtocolMessage(role="bot", content="Token counting is the process of..."),
        fp.ProtocolMessage(role="user", content="Can you provide an example?"),
        fp.ProtocolMessage(
            role="bot", content="Sure! Here's an example of token counting..."
        ),
    ]

    result = count_message_tokens(messages)

    # System and user messages should be in prompt_tokens
    prompt_token_count = (
        count_tokens("You are a helpful AI assistant.")
        + count_tokens("Tell me about token counting.")
        + count_tokens("Can you provide an example?")
        + 3  # overhead
    )

    # Bot messages should be in completion_tokens
    completion_token_count = count_tokens(
        "Token counting is the process of..."
    ) + count_tokens("Sure! Here's an example of token counting...")

    assert result["prompt_tokens"] == prompt_token_count
    assert result["completion_tokens"] == completion_token_count
    assert result["total_tokens"] == prompt_token_count + completion_token_count


def test_count_message_tokens_null_content():
    """Test counting tokens with empty content"""
    messages = [
        fp.ProtocolMessage(role="user", content=""),
        fp.ProtocolMessage(role="bot", content=""),
    ]

    result = count_message_tokens(messages)

    # Should only have the overhead tokens for prompt and empty strings for content
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 3


def test_consistent_tokenization():
    """Test that tokenization is consistent regardless of model parameter"""
    text = "This is a test message with some tokens."

    # Count tokens with different model params - all should be the same
    count1 = count_tokens(text)
    count2 = count_tokens(text, model="GPT-4o")
    count3 = count_tokens(text, model="Claude-3.5-Sonnet")

    assert count1 == count2 == count3
