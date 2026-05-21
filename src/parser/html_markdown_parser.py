from html.parser import HTMLParser

class HTMLToMarkdownParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ["p", "div"]:
            self.parts.append("\n")
        elif tag in ["h1", "h2", "h3", "h4"]:
            self.parts.append("\n## ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "strong":
            self.parts.append("**")
        elif tag == "em":
            self.parts.append("*")
        elif tag == "code":
            self.parts.append("`")

    def handle_endtag(self, tag):
        if tag == "strong":
            self.parts.append("**")
        elif tag == "em":
            self.parts.append("*")
        elif tag == "code":
            self.parts.append("`")
        elif tag in ["p", "ul", "ol"]:
            self.parts.append("\n")

    def handle_data(self, data):
        cleaned = data.strip()
        if not cleaned:
            return
        BAD_STRINGS = {
            "text",
            "label",
            "['text', 'label']",
            '["text", "label"]'
        }
        if cleaned.lower() in BAD_STRINGS:
            return
        self.parts.append(cleaned)

    def get_markdown(self):
        return "".join(self.parts).strip()
