from collections import defaultdict


class RelationStore:

    def __init__(self):

        self.relations = defaultdict(
            lambda: {
                "weight": 0,
                "contexts": set()
            }
        )

    def add_relation(self, source, relation, target, context="conversation"):
        source = source.strip().lower()
        target = target.strip().lower()

        if not source or not target:
            return

        key = (source, relation, target)
        self.relations[key]["weight"] += 1
        self.relations[key]["contexts"].add(context)

    def get_relations(self):
        output = []
        for (source, relation, target), value in self.relations.items():
            output.append({
                "source": source,
                "relation": relation,
                "target": target,
                "weight": value["weight"],
                "contexts": list(
                    value["contexts"]
                )
            })
        return output