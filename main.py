from pathlib import Path
from src.parser.gemini import GeminiChatParser
from src.utils.obsidian_enricher import ObsidianEnricher

VAULT_ROOT = Path("C:/Users/Suryansh/Documents/Personal/KnowledgeBase/KnowledgeBase")

CONVERSATION_PATH = VAULT_ROOT / "conversations" / "gemini"

def main():

    parser = GeminiChatParser(
        input_file=Path("data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"),
        output_dir=CONVERSATION_PATH
    )
    parser.parse()

    enricher = ObsidianEnricher(vault_dir=CONVERSATION_PATH)
    enricher.enrich()

if __name__ == "__main__":
    main()