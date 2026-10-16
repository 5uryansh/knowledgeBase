from knowledgebase.graph.analytics import compute_analytics
from knowledgebase.graph.cooccurrence import CooccurrenceBuilder
from knowledgebase.graph.model import KnowledgeGraph
from knowledgebase.graph.relations import RelationExtractor
from knowledgebase.graph.specificity import specificity


def test_npmi_penalises_hub_nodes():
    b = CooccurrenceBuilder()
    # "me" appears everywhere (a hub); "triton" and "softmax" only ever appear together.
    for i in range(10):
        b.add_conversation(f"c{i}", [f"[[me|I]] talked about [[topic{i}|topic {i}]]."])
    b.add_conversation("x", ["[[me|I]] wrote a [[triton|Triton]] [[softmax|softmax]] kernel."])
    b.add_conversation("y", ["[[triton|Triton]] fused [[softmax|softmax]] again."])
    edges = {(e["source"], e["target"]): e for e in b.export()}
    specific = edges[("softmax", "triton")]
    hub = edges[("me", "triton")]
    assert specific["npmi"] > hub["npmi"]
    assert specific["strength"] > hub["strength"]


def test_sentence_cooccurrence_outweighs_paragraph():
    b = CooccurrenceBuilder()
    b.add_conversation("c", ["[[a|A]] and [[b|B]] here. Then [[c|C]] later."])
    edges = {(e["source"], e["target"]): e for e in b.export()}
    assert edges[("a", "b")]["sentence"] == 1
    assert edges[("a", "c")]["sentence"] == 0 and edges[("a", "c")]["paragraph"] == 1


def test_relations_use_nearest_nodes_around_the_trigger():
    r = RelationExtractor()
    r.process("[[vllm|vLLM]] with [[docker|Docker]] relies on [[pagedattention|PagedAttention]] for memory.", "c1")
    rels = {(x["source"], x["relation"], x["target"]) for x in r.export()}
    assert ("docker", "DEPENDS_ON", "pagedattention") in rels
    assert not any(s == "vllm" and t == "docker" for s, _, t in rels)
    assert r.export()[0]["evidence"][0].startswith("vLLM with Docker")


def test_specificity_prefers_rare_terms():
    assert specificity("vLLM", "entity", "technology") > specificity("model")
    assert specificity("partial refresh") > specificity("update")


def test_analytics_and_queries():
    nodes = {n: {"id": n, "label": n.upper(), "kind": "entity", "type": "technology", "conversation_count": 3}
             for n in ["a", "b", "c", "d", "e", "f"]}
    edges = [
        {"source": "a", "target": "b", "strength": 2.0, "npmi": 0.8, "conversations": 3},
        {"source": "b", "target": "c", "strength": 2.0, "npmi": 0.8, "conversations": 3},
        {"source": "a", "target": "c", "strength": 2.0, "npmi": 0.8, "conversations": 3},
        {"source": "c", "target": "d", "strength": 0.3, "npmi": 0.2, "conversations": 2},
        {"source": "d", "target": "e", "strength": 2.0, "npmi": 0.8, "conversations": 3},
        {"source": "e", "target": "f", "strength": 2.0, "npmi": 0.8, "conversations": 3},
        {"source": "d", "target": "f", "strength": 2.0, "npmi": 0.8, "conversations": 3},
    ]
    g = compute_analytics(KnowledgeGraph(nodes=nodes, edges=edges))
    assert len(g.communities) == 2
    assert {"importance", "pagerank", "community", "betweenness"} <= set(g.nodes["a"])
    assert g.shortest_path("a", "f")[0] == "a" and g.shortest_path("a", "f")[-1] == "f"
    assert {n["id"] for n in g.search_nodes("A")} >= {"a"}
    # c and d bridge the two triangles
    assert g.nodes["c"]["betweenness"] > g.nodes["a"]["betweenness"]
