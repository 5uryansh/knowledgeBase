import re
from gliner import GLiNER
import yaml
from pathlib import Path


STOP_ENTITIES = {
    "gemini",
    "json",
    "markdown",
    "response",
    "prompt",
    "timestamp",
    "platform",
    "source",
    "hash"
}


class EntityExtractor:

    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

        config_path = Path("config/entity_labels.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.labels = config["labels"]
        self.regex_patterns = [
            r'\b[A-Z][a-zA-Z0-9\-\_]+\b',
            r'\b(?:GPT|Gemma|Llama|BERT)\-?[A-Za-z0-9\.]*\b',
            r'\b(?:PyTorch|TensorFlow|Neo4j|CUDA|PostgreSQL)\b',
            r'\b[a-zA-Z_]+\.[a-zA-Z_]+\b'
        ]

    def clean_text(self, text):
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _valid_entity(self, entity):
        entity = entity.strip()
        if len(entity) < 3:
            return False
        if entity.lower() in STOP_ENTITIES:
            return False
        if entity.isdigit():
            return False
        return True

    def extract_entities(self, text):
        text = self.clean_text(text)
        entities = set()
        predictions = self.model.predict_entities(text, self.labels)

        for prediction in predictions:
            entity = prediction["text"].strip()

            if not self._valid_entity(entity):
                continue

            entities.add(entity)

        for pattern in self.regex_patterns:
            matches = re.findall(pattern, text)

            for match in matches:
                match = match.strip()
                if not self._valid_entity(match):
                    continue
                entities.add(match)

        return sorted(entities)