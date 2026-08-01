<img width="2048" height="1280" alt="image" src="https://github.com/user-attachments/assets/42969e9c-f2d6-4989-bb2e-e13a970e0eb3" />

# 🧠 KnowledgeBase

> **A Graph-first Personal Cognition System**
>
> Building an AI that doesn't just retrieve information—but understands how ideas connect, evolve, and relate over time.

---

## Motivation

Every day we interact with multiple AI systems:

- ChatGPT
- Claude
- Gemini
- OpenCode
- Terminal sessions
- Documentation
- GitHub

Each conversation contains knowledge.

Unfortunately, that knowledge is scattered.

Finding *where* something was discussed is often harder than understanding it.

This project aims to solve that problem.

Instead of treating conversations as independent chat logs, the system transforms them into an interconnected knowledge graph capable of semantic retrieval, graph reasoning, and eventually autonomous knowledge discovery.

---

# Vision

Instead of this...

```
ChatGPT
 ├── Conversation A
 ├── Conversation B
 └── Conversation C

Claude
 ├── Conversation D
 └── Conversation E

Gemini
 ├── Conversation F
```

we build...

```
                     CUDA
                    /    \
              PyTorch   Triton
                 |         |
                 |      Quantization
                 |
    Embeddings ----- Retrieval ----- FAISS
          |                |
          |             GraphRAG
          |
     World Models
          |
     Reinforcement Learning
```

Knowledge becomes connected by meaning rather than by conversation boundaries.

---

# Architecture

```
                  Raw AI Exports
        ┌────────────┬─────────────┐
        │            │             │
    ChatGPT      Claude        Gemini
        │            │             │
        └────────────┴─────────────┘
                     │
              Conversation Parsers
                     │
             Standard Markdown
                     │
            Obsidian Enrichment
        ┌────────────┴─────────────┐
        │                          │
   Topic Extraction         Entity Extraction
        │                          │
        └────────────┬─────────────┘
                     │
              Wiki-style Linking
                     │
      [[CUDA]] [[PyTorch]] [[FAISS]]
                     │
          ┌──────────┴──────────┐
          │                     │
   Knowledge Graph        Mention Index
          │
   Edge Extraction
          │
   Graph Export (JSON)
          │
──────────┼────────────────────────────────────────────
          │
      Chunking
          │
     Embedding Model
          │
      FAISS Index
          │
  Graph-aware Retriever
          │
    Future LLM Interface
```

---

# Current Pipeline

## 1. Conversation Parsing

Every supported AI platform is converted into a unified markdown representation.

Currently supported:

- ChatGPT
- Claude
- Gemini

Regardless of the original export format, every conversation becomes a standardized markdown document.

---

## 2. Markdown Enrichment

Each conversation is processed through an enrichment pipeline.

### Topic Extraction

Automatically detects high-level discussion topics.

Example

```
Diffusion Models

↓

[[diffusion_models]]
```

---

### Entity Extraction

Named entities are extracted using GLiNER.

Examples

```
CUDA
PyTorch
Prisma
Zero Sync
FAISS
```

---

### Wiki Link Injection

Entities and topics are automatically converted into Obsidian links.

Example

```
CUDA

↓

[[cuda|CUDA]]
```

This enables:

- backlink generation
- graph visualization
- mention tracking

---

### Mention Writer

Every entity maintains its own page.

Example

```
entities/

CUDA.md

PyTorch.md

FAISS.md
```

Each entity page automatically receives backlinks to conversations mentioning it.

---

# Knowledge Graph

The graph is built entirely from markdown.

Every paragraph and sentence contributes relationships.

Example

```
CUDA
PyTorch
Embeddings
```

becomes

```
CUDA ───── PyTorch
   │
   │
Embeddings
```

Edges are weighted according to context.

Example

| Context | Weight |
|----------|--------|
| Sentence | High |
| Paragraph | Medium |
| Conversation | Low |

This creates a weighted co-occurrence graph instead of simple hyperlinks.

---

# Relation Extraction

Beyond co-occurrence, explicit relationships are extracted.

Example

```
CUDA accelerates PyTorch

↓

CUDA
   ── accelerates ──►
                     PyTorch
```

These relations are exported separately for future reasoning.

---

# Retrieval Pipeline

The current retriever combines semantic search with graph information.

```
User Query
      │
      ▼
Embedding Model
      │
      ▼
Vector Search (FAISS)
      │
      ▼
Entity Detection
      │
      ▼
Graph Expansion
      │
      ▼
Chunk Retrieval
```

The graph is not replacing vector search.

It augments it.

---

# Current Indexes

```
indexes/

embeddings.faiss

chunk_ids.txt

chunks.json

entity_index.json

graph_edges.json

graph_relations.json
```

Each file has a dedicated responsibility.

---

## chunks.json

Stores every chunk.

```
Chunk
├── id
├── source
├── links
└── text
```

---

## embeddings.faiss

Stores vector embeddings.

Used for semantic retrieval.

---

## entity_index.json

Maps normalized entities to canonical nodes.

Example

```
CUDA

↓

cuda
```

---

## graph_edges.json

Stores the weighted co-occurrence graph.

Example

```
CUDA

↓

PyTorch

↓

Embeddings
```

---

## graph_relations.json

Stores explicit semantic relationships.

---

# Obsidian

The vault is designed to be human-readable.

```
KnowledgeBase/

conversations/

entities/

topics/

indexes/
```

Conversations remain readable while simultaneously acting as machine-processable knowledge.

---

# Why Obsidian?

Obsidian gives us

- backlinks
- graph visualization
- local-first storage
- markdown interoperability

The vault acts as both

- human interface
- machine knowledge base

---

# Current Retrieval Strategy

```
                  Query
                    │
        ┌───────────┴────────────┐
        │                        │
   Vector Retrieval         Entity Match
        │                        │
        ▼                        ▼
 Relevant Chunks          Seed Entities
        │                        │
        └───────────┬────────────┘
                    ▼
             Graph Expansion
                    │
                    ▼
          Expanded Context
```

This architecture combines semantic similarity with structural knowledge.

---

# Design Principles

## Graph First

Knowledge is represented as relationships rather than isolated documents.

---

## Local First

Everything remains local.

No cloud dependency.

---

## Platform Agnostic

The parser normalizes every AI platform into the same internal representation.

Adding a new provider should require only a parser.

---

## Extensible

Every major subsystem is independent.

```
Parser

↓

Markdown

↓

Graph

↓

Retriever

↓

LLM
```

Each layer can evolve without affecting the others.

---

# Roadmap

## Retrieval

- Hybrid Retrieval
- Better Graph Expansion
- Cross-Encoder Reranking
- Multi-hop Retrieval

---

## Evaluation

A proper benchmark suite will be built.

Rather than manually inspecting retrieval quality, the system will automatically evaluate itself using synthetic benchmark questions generated from indexed chunks.

Metrics will include:

- Recall@K
- MRR
- Latency
- Ablation studies
- Failure categorization

This enables objective comparison between retrieval strategies.

---

## Knowledge Graph

Future work includes:

- Community Detection
- Temporal Graphs
- Entity Canonicalization
- Knowledge Evolution
- Graph Analytics

---

## LLM

Eventually the retriever will power an LLM capable of:

- citing sources
- long-term memory
- graph-aware reasoning
- multi-hop exploration
- personal research assistance

---

# Long-Term Goal

The objective is not to build another chatbot.

The objective is to build a persistent cognitive system that continuously grows with every conversation.

Instead of forgetting after every interaction, the system accumulates knowledge, connects ideas across time, and makes past understanding immediately accessible.

Over time, the knowledge graph itself becomes a map of learning, revealing not only what is known, but how concepts relate, evolve, and influence one another.