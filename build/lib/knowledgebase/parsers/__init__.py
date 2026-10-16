"""Chat export parsers. Each one yields :class:`~knowledgebase.parsers.base.Conversation`."""
from knowledgebase.parsers.base import Conversation, Message
from knowledgebase.parsers.chatgpt import ChatGPTChatParser
from knowledgebase.parsers.claude import ClaudeChatParser
from knowledgebase.parsers.claude_code import ClaudeCodeChatParser
from knowledgebase.parsers.gemini import GeminiChatParser

PARSERS = {
    "chatgpt": ChatGPTChatParser,
    "claude": ClaudeChatParser,
    "gemini": GeminiChatParser,
    "claude-code": ClaudeCodeChatParser,
}

__all__ = [
    "Conversation",
    "Message",
    "PARSERS",
    "ChatGPTChatParser",
    "ClaudeChatParser",
    "ClaudeCodeChatParser",
    "GeminiChatParser",
]
