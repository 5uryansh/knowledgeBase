import re
from pathlib import Path

def normalize_node_name(text):
    text = text.strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    return text.lower()

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '', text).strip()

def find_markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))