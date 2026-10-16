import json

from conftest import SAMPLES

from knowledgebase.parsers import ChatGPTChatParser, ClaudeChatParser, ClaudeCodeChatParser, GeminiChatParser
from knowledgebase.parsers.html_markdown import html_to_markdown


def test_chatgpt_parser_reads_sample_export():
    convs = list(ChatGPTChatParser(SAMPLES / "chatgpt" / "conversations.json").parse())
    assert len(convs) == 13
    first = convs[0]
    assert first.platform == "chatgpt" and first.created_at is not None
    assert [m.role for m in first.messages[:2]] == ["user", "assistant"]


def test_chatgpt_parser_follows_active_branch(tmp_path):
    # An edited prompt creates two branches; only the current_node branch should be kept.
    mapping = {
        "root": {"id": "root", "message": None, "parent": None, "children": ["a"]},
        "a": {"id": "a", "parent": "root", "children": ["b1", "b2"], "message": {
            "author": {"role": "user"}, "create_time": 1, "content": {"content_type": "text", "parts": ["question"]}}},
        "b1": {"id": "b1", "parent": "a", "children": [], "message": {
            "author": {"role": "assistant"}, "create_time": 2, "content": {"content_type": "text", "parts": ["old answer"]}}},
        "b2": {"id": "b2", "parent": "a", "children": [], "message": {
            "author": {"role": "assistant"}, "create_time": 3, "content": {"content_type": "text", "parts": ["new answer"]}}},
    }
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([{"title": "t", "create_time": 1, "mapping": mapping, "current_node": "b2", "id": "x"}]))
    (conv,) = ChatGPTChatParser(path).parse()
    assert [m.text for m in conv.messages] == ["question", "new answer"]


def test_claude_parser():
    convs = list(ClaudeChatParser(SAMPLES / "claude" / "conversations.json").parse())
    assert len(convs) == 11
    assert all(c.messages[0].role == "user" for c in convs)


def test_gemini_parser_groups_by_day_and_converts_html():
    convs = list(GeminiChatParser(SAMPLES / "gemini" / "MyActivity.json").parse())
    assert len(convs) == 7
    assert not any(m.text.startswith("Prompted ") for c in convs for m in c.messages)


def test_claude_code_parser_uses_summary_as_title():
    convs = list(ClaudeCodeChatParser(SAMPLES / "claude-code" / "projects").parse())
    assert len(convs) == 5
    assert any(c.title.startswith("esp32-weather firmware") for c in convs)
    assert all(c.project for c in convs)


def test_uid_is_stable_across_reimports():
    a = [c.uid for c in ClaudeChatParser(SAMPLES / "claude" / "conversations.json").parse()]
    b = [c.uid for c in ClaudeChatParser(SAMPLES / "claude" / "conversations.json").parse()]
    assert a == b and len(set(a)) == len(a)


def test_html_to_markdown_never_emits_turn_headings():
    md = html_to_markdown("<h2>Title</h2><p>Some <strong>bold</strong> text</p><ul><li>one</li></ul>")
    assert "## " not in md
    assert "**bold**" in md and "- one" in md
