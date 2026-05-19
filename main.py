from pathlib import Path
from src.parser.gemini import GeminiChatParser

def main():

    parser = GeminiChatParser(
        input_file=Path(
            "data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json"
        ),
        output_dir=Path(
            "/mnt/c/Users/Suryansh/Documents/Personal/KnowledgeBase/KnowledgeBase/gemini"
        )
    )

    parser.parse()

if __name__ == "__main__":
    main()