from pathlib import Path
from src.parser.gemini import GeminiChatParser
from src.utils.obsidian_enricher import ObsidianEnricher

VAULT_PATH = Path("/mnt/c/Users/Suryansh/Documents/Personal/KnowledgeBase/KnowledgeBase/gemini")

def main():

    parser = GeminiChatParser(
        input_file=Path("data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"),
        output_dir=VAULT_PATH
    )
    parser.parse()

    enricher = ObsidianEnricher(vault_dir=VAULT_PATH)
    enricher.enrich()

if __name__ == "__main__":
    main()