"""Backwards-compatible entry point: `python main.py` == `kb build`.

The pipeline now lives in the `knowledgebase` package (see README); configure paths in kb.yaml.
"""
import sys

from knowledgebase.cli import main

if __name__ == "__main__":
    sys.exit(main(["build", *sys.argv[1:]]))
