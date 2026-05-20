import re
from pathlib import Path
from src.utils.topic_extractor import TopicExtractor
from src.utils.entity_extractor import EntityExtractor
from tqdm import tqdm

class ObsidianEnricher:

    def __init__(self, vault_dir):
        self.vault_dir = Path(vault_dir)
        self.topic_extractor = TopicExtractor()
        self.entity_extractor = EntityExtractor()

        self.entities_dir = self.vault_dir.parent.parent / "entities"
        self.topics_dir = self.vault_dir.parent.parent / "topics"

        self.entities_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)

    def _load_markdown_files(self):
        markdown_files = list(
            self.vault_dir.rglob("*.md")
        )
        documents = []
        for md_file in markdown_files:
            with open(md_file, "r", encoding="utf-8") as f:
                documents.append(f.read())
        return markdown_files, documents

    def _sanitize_filename(self, text):
        return re.sub(r'[<>:"/\\|?*]', '', text).strip()

    def _create_entity_notes(self, entities):
        for entity in entities:
            safe_entity = self._sanitize_filename(entity)
            path = self.entities_dir / f"{safe_entity}.md"
            if not path.exists():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {entity}\n")

    def _create_topic_notes(self, topics):
        for topic in topics:
            safe_topic = self._sanitize_filename(topic)
            path = self.topics_dir / f"{safe_topic}.md"
            if not path.exists():
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {topic}\n")

    def _inject_links(self, text, topics, entities):
        all_links = set(topics + entities)
        for item in all_links:
            escaped = re.escape(item)
            pattern = rf'(?<!\[\[)\b{escaped}\b(?!\]\])'
            replacement = lambda m: f"[[{m.group(0)}]]"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def enrich(self):
        markdown_files, documents = self._load_markdown_files()
        
        for md_file, document in tqdm(zip(markdown_files, documents), total=len(markdown_files), desc="Enriching markdown files"):
            topics = self.topic_extractor.extract_topics([document])
            entities = self.entity_extractor.extract_entities(document)

            self._create_entity_notes(entities)
            self._create_topic_notes(topics)

            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            enriched_content = self._inject_links(content, topics, entities)

            with open(md_file, "w", encoding="utf-8") as f:
                f.write(enriched_content)