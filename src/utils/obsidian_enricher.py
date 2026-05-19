import re
from pathlib import Path
from src.utils.topic_extractor import TopicExtractor
from tqdm import tqdm

class ObsidianEnricher:

    def __init__(self, vault_dir):
        self.vault_dir = Path(vault_dir)
        self.extractor = TopicExtractor()

    def _load_markdown_files(self):
        markdown_files = list(
            self.vault_dir.rglob("*.md")
        )
        documents = []
        for md_file in markdown_files:
            with open(md_file, "r", encoding="utf-8") as f:
                documents.append(f.read())
        return markdown_files, documents

    def _inject_links(self, text, topics):
        for topic in topics:
            escaped = re.escape(topic)
            pattern = rf'(?<!\[\[)\b{escaped}\b(?!\]\])'
            replacement = lambda m: f"[[{m.group(0)}]]"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def enrich(self):
        markdown_files, documents = self._load_markdown_files()
        
        for md_file, document in tqdm(
            zip(markdown_files, documents),
            total=len(markdown_files),
            desc="Enriching markdown files"
        ):
            topics = self.extractor.extract_topics([document])

            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            enriched_content = self._inject_links(content, topics)

            with open(md_file, "w", encoding="utf-8") as f:
                f.write(enriched_content)

            print(f"Enriched: {md_file}")