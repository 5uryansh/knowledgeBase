import re

def normalize_node_name(text):
    text = text.strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    return text.lower()

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '', text).strip()