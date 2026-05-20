import re
from collections import Counter
import yake
import spacy


STOP_TOPICS = {
    "response",
    "data",
    "string",
    "prompt",
    "text",
    "output",
    "input",
    "file",
    "code",
    "message",
    "result",
    "information",
    "question",
    "answer",
    "content",
    "label",
    "html",
    "json",
    "markdown",
    "gemini",
    "timestamp",
    "platform",
    "source",
    "hash",
    "title"
}


class TopicExtractor:

    def __init__(self):
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.3,
            top=5
        )
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'import .*', '', text)
        text = re.sub(r'from .* import .*', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[#>*_`-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _valid_topic(self, topic):
        topic = topic.strip().lower()

        if topic in STOP_TOPICS:
            return False

        if len(topic) < 4:
            return False

        if len(topic.split()) > 3:
            return False
        if topic.isdigit():
            return False
        if not any(c.isalpha() for c in topic):
            return False
        if re.search(r'\d', topic):
            return False
        if any(char in topic for char in "{}[]()/\\=:;<>@#$%^&*|"):
            return False
        if topic.startswith(("the ", "this ", "that ")):
            return False

        return True

    def extract_topics(self, documents):
        cleaned_docs = [
            self.clean_text(doc)
            for doc in documents
        ]

        topic_counter = Counter()

        for doc in cleaned_docs:

            yake_keywords = self.keyword_extractor.extract_keywords(doc)

            for keyword, score in yake_keywords:
                keyword = keyword.lower().strip()
                keyword = re.sub(r'\s+', ' ', keyword)

                if not self._valid_topic(keyword):
                    continue

                topic_counter[keyword.title()] += 2

            parsed_doc = self.nlp(doc)

            for chunk in parsed_doc.noun_chunks:
                topic = chunk.text.lower().strip()
                topic = re.sub(r'\s+', ' ', topic)

                if not self._valid_topic(topic):
                    continue

                topic_counter[topic.title()] += 1

        final_topics = []

        for topic, count in topic_counter.most_common():

            if count < 5:
                continue

            final_topics.append(topic)

        return final_topics[:25]