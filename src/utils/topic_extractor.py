import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

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
    "content"
}

class TopicExtractor:

    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            verbose=True,
            calculate_probabilities=False
        )

    def clean_text(self, text):
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'[#>*_`-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_topics(self, documents):
        cleaned_docs = [
            self.clean_text(doc)
            for doc in documents
        ]
        topics, _ = self.topic_model.fit_transform(
            cleaned_docs
        )
        topic_info = self.topic_model.get_topic_info()
        final_topics = set()
        
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue
            top_keyword = topic_words[0][0].strip().lower()
            if top_keyword in STOP_TOPICS:
                continue
            if len(top_keyword) < 4:
                continue
            if top_keyword.isdigit():
                continue
            final_topics.add(top_keyword)

        return sorted(final_topics)