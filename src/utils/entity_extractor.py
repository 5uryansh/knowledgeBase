import re
from collections import Counter
from gliner import GLiNER
import yaml
from pathlib import Path


config_path = Path("config/stop_words.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    
STOP_ENTITIES = config["words"]


class EntityExtractor:

    def __init__(self):
        self.model = GLiNER.from_pretrained(
            "urchade/gliner_small-v2.1"
        )

        config_path = Path("config/entity_labels.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.labels = config["labels"]

        self.regex_patterns = [
            r'\b[A-Z][a-zA-Z0-9\-\_]+\b',
            r'\b(?:GPT|Gemma|Llama|BERT)\-?[A-Za-z0-9\.]*\b',
            r'\b[a-zA-Z_]+\.[a-zA-Z_]+\b'
        ]

    def clean_text(self, text):
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _normalize_entity(self, entity):
        entity = entity.strip()
        entity = re.sub(r'\s+', ' ', entity)
        return entity

    def _valid_entity(self, entity):
        entity = entity.strip()

        if len(entity) < 3:
            return False

        if len(entity.split()) > 4:
            return False

        if entity.lower() in STOP_ENTITIES:
            return False

        if entity.isdigit():
            return False

        if not any(c.isalpha() for c in entity):
            return False

        return True

    def extract_entities(self, text):
        text = self.clean_text(text)

        entity_counter = Counter()

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1200:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        for chunk in chunks:
            predictions = self.model.predict_entities(chunk, self.labels)

            for prediction in predictions:
                entity = self._normalize_entity(prediction["text"])

                if not self._valid_entity(entity):
                    continue

                entity_counter[entity] += 2

            for pattern in self.regex_patterns:
                matches = re.findall(pattern, chunk)

                for match in matches:
                    entity = self._normalize_entity(match)

                    if not self._valid_entity(entity):
                        continue

                    entity_counter[entity] += 1

        final_entities = []

        for entity, count in entity_counter.most_common():
            if count < 2:
                continue

            final_entities.append(entity)

        return final_entities[:50]