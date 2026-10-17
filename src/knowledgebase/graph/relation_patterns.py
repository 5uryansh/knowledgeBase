"""Trigger phrases for typed relations. Extend freely — keys become edge labels."""

RELATION_PATTERNS = {
    "USES": ["uses", "utilizes", "runs on", "powered by", "built with", "built on", "is used in", "used with"],
    "DEPENDS_ON": ["depends on", "requires", "relies on", "needs"],
    "IMPLEMENTS": ["implements", "provides", "supports"],
    "OPTIMIZES": ["optimizes", "improves", "accelerates", "speeds up"],
    "CAUSES": ["causes", "leads to", "results in"],
    "COMPETES_WITH": ["competes with", "alternative to", "instead of", "versus", "vs"],
    "PART_OF": ["part of", "component of", "inside", "belongs to"],
    "REPLACES": ["replaces", "replaced", "migrated from", "switched from"],
    "IS_A": ["is a type of", "is a kind of", "is an example of"],
}
