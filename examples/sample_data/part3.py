"""Sample conversations, part 3: DevOps, reinforcement learning / world models, agents and MCP."""

CONVERSATIONS = [
    ("chatgpt", "2025-11-15", "Docker Compose for a FastAPI + PostgreSQL stack", [
        "Write a docker-compose.yml for FastAPI, PostgreSQL and Caddy as a reverse proxy with automatic HTTPS.",
        "Three services: api (built from your Dockerfile, running uvicorn), db (postgres:16 with a named volume and a healthcheck using pg_isready) and caddy (caddy:2 with a Caddyfile that reverse_proxies to api:8000). The api service depends_on db with condition: service_healthy. Caddy obtains Let's Encrypt certificates automatically when the domain points at the server.",
        "How do I keep secrets out of the compose file?",
        "Use an .env file that Compose reads automatically, referenced as ${POSTGRES_PASSWORD}, and keep it out of git. For production on a single VM, Docker secrets or a secrets manager are better; at minimum restrict file permissions.",
    ]),
    ("claude", "2025-12-28", "Oracle Cloud free tier ARM instance for self-hosting", [
        "Can I self-host my side projects on the Oracle Cloud free tier ARM instance?",
        "Yes, the Always Free Ampere A1 shape gives up to 4 OCPUs and 24 GB RAM, which is plenty for Docker Compose stacks, a Postgres database and small services. Capacity is often unavailable in popular regions, so people script retries of instance creation. Use ARM64 images: most official Docker images are multi-arch, but check custom ones. Open ports both in the VCN security list and in the instance firewall.",
        "Can it run an LLM?",
        "CPU-only. With llama.cpp and a 4-bit GGUF quantized 7B model you can get a few tokens per second on 4 Ampere cores, fine for background jobs but slow for chat. For interactive use, a GPU host or a hosted inference endpoint is better.",
    ]),
    ("gemini", "2026-01-30", "GitHub Actions CI for a Python project", [
        "Set up GitHub Actions to run pytest and ruff on every pull request.",
        "Create .github/workflows/ci.yml triggered on pull_request and push to main. Use actions/checkout, actions/setup-python with cache: pip, install with pip install -e .[dev], then run ruff check . and pytest -q. Add a matrix over Python 3.10-3.12 if you support several versions. Protect main so the CI check is required before merging.",
    ]),
    ("claude", "2026-04-02", "Kubernetes or just Docker Compose?", [
        "Should I move my side projects from Docker Compose to Kubernetes?",
        "Probably not. Kubernetes shines with many services, autoscaling, multiple nodes and teams. For a few services on one VM, Docker Compose is simpler and cheaper to operate. If you want a middle ground, k3s gives you lightweight Kubernetes on a single node, but you take on manifests, ingress controllers and certificate management. Move when Compose's limitations actually hurt.",
    ]),
    ("chatgpt", "2025-10-28", "PPO explained: proximal policy optimization", [
        "Explain PPO for reinforcement learning without too much math.",
        "PPO is a policy-gradient method. The agent collects trajectories with the current policy, estimates advantages (how much better an action was than expected, usually with GAE), and updates the policy to make good actions more likely. The key trick is the clipped objective: it limits how far the new policy can move from the old one in a single update, which keeps training stable. PPO is popular because it is simple, robust and works for both discrete and continuous actions.",
        "Why is my PPO agent's reward collapsing after improving?",
        "Common causes: learning rate too high, too many epochs per batch, or no advantage normalization. Track the approximate KL divergence and the clip fraction; if KL spikes, lower the learning rate or add early stopping on KL. Also check the entropy: if it collapses, the policy stops exploring.",
    ]),
    ("claude", "2025-11-22", "World models: Dreamer and learning in imagination", [
        "What are world models in reinforcement learning, like DreamerV3?",
        "A world model learns the environment's dynamics in a compact latent space: an encoder maps observations to latent states, a recurrent model predicts the next latent state and the reward, and a decoder reconstructs observations. DreamerV3 then trains the actor and critic entirely on imagined trajectories inside the world model, which makes it far more sample-efficient than model-free methods like PPO. Model-based RL trades environment interactions for compute.",
        "How is that related to embeddings?",
        "The latent state is a learned embedding of the environment, like a sentence embedding is a learned representation of text. Both compress high-dimensional inputs into vectors where similar situations are close. In world models the representation is trained to be predictive of the future rather than to match similar texts.",
        "And MuZero?",
        "MuZero also learns a model, but it never reconstructs observations: its latent dynamics only need to predict reward, value and policy, and it plans with Monte Carlo tree search in that latent space. Dreamer imagines with a learned simulator; MuZero plans with a learned value-equivalent model.",
    ]),
    ("gemini", "2025-12-03", "Diffusion models vs transformers for video world models", [
        "Are new video world models diffusion models or transformers?",
        "Both, often combined: a diffusion transformer (DiT) uses a transformer backbone to denoise latent video patches. Action-conditioned versions predict the next frames given an action, which turns a video generator into an interactive world model for agents and robotics. Training such models needs large GPU clusters and efficient attention kernels like FlashAttention.",
    ]),
    ("claude", "2026-04-20", "Building an MCP server for my knowledge graph", [
        "I want my coding agents to query my personal knowledge graph. Should I build an MCP server?",
        "Yes. The Model Context Protocol lets any MCP client (Claude Code, Claude Desktop, Cursor, OpenCode) discover and call your tools. Expose a few focused tools: search_knowledge for graph-aware retrieval, explore_concept for a node and its neighbors, and find_connection for paths between two concepts. Use the Streamable HTTP transport so one local server on a port serves every client; JSON-RPC requests go to a single endpoint like /api/mcp.",
        "How do I make the tool results useful for the model?",
        "Return compact text with the conversation title, date and source so the model can cite, plus structured content for programmatic use. Keep results short (top 5-10 passages), explain why each result matched (which concepts connected it) and handle unknown concepts with a clear error message instead of an exception.",
        "Security concerns?",
        "Bind to localhost by default and validate the Origin header to prevent DNS rebinding attacks from browsers. Tools here are read-only, which you can declare with the readOnlyHint annotation.",
    ]),
    ("claude-code", "2026-05-06", "knowledge-graph repo: benchmark retrieval with Recall@K and MRR", [
        "Add a benchmark to the knowledge graph repo that compares BM25, dense, hybrid and graph retrieval.",
        "Added benchmarks/run.py: it builds an index per corpus, runs each question through every retrieval mode and reports Recall@5, Recall@10, MRR and nDCG@10 by question type. Graph retrieval uses Personalized PageRank seeded by the query's concepts and fused with BM25 and dense rankings through reciprocal rank fusion. Parameters are tuned on a dev split and reported on a held-out test split.",
        "Graph mode is worse than BM25 on single-hop questions.",
        "Expected: pure graph retrieval only knows concepts, so for exact lookups BM25 wins. That is why the default mode fuses all three signals; the graph contributes most on multi-hop questions. I also added IDF weighting so hub concepts no longer receive seed mass, which removed most of the noise.",
    ]),
    ("chatgpt", "2026-05-18", "OpenAI-compatible API on a Hugging Face Space", [
        "Can I host an open model on a Hugging Face Space and expose an OpenAI-compatible API for agents?",
        "Yes. A Gradio or FastAPI Space can wrap the model and expose /v1/chat/completions. On ZeroGPU Spaces the GPU is attached per request, so cold starts add latency and long-running servers like vLLM are not a fit; use transformers generate inside the request handler instead. For agents that need tool calling, return the OpenAI tool_calls format so frameworks can parse it.",
        "Rate limits on the free tier?",
        "ZeroGPU has a daily GPU-time quota per user, so heavy agent loops will hit it. Cache responses where possible and use a smaller quantized model for routine tool calls.",
    ]),
    ("claude-code", "2026-06-10", "knowledge-graph repo: lazy-loading graph explorer UI", [
        "Build a graph explorer UI that doesn't load all nodes at once.",
        "The UI now requests /api/graph/overview, which returns only the top nodes by importance (PageRank blended with how many conversations mention the concept), diversified across communities. Clicking a node calls /api/graph/node/{id} and merges that neighborhood into the force-directed graph, so the browser only ever holds what you explored. The side panel shows related concepts, typed relations with evidence, the conversations that mention the node and a timeline.",
        "Add something that helps discover new things.",
        "Added a Discover panel computed from the graph structure: bridge concepts with high betweenness that connect different communities, unexpected links between communities with high NPMI, hidden connections (concept pairs never discussed together but sharing many neighbors, scored with Adamic–Adar) and forgotten topics you haven't mentioned in months.",
    ]),
    ("gemini", "2026-07-02", "Tool calling with small open models", [
        "Which small open models are good at tool calling for agents?",
        "Qwen2.5 7B Instruct and Llama 3.1 8B Instruct handle function calling reasonably well, especially with a strict JSON schema and few tools. Quantized 4-bit versions keep most of that ability. Serve them with vLLM or llama.cpp using an OpenAI-compatible API so agent frameworks work unchanged, and keep tool descriptions short and explicit.",
    ]),
    ("chatgpt", "2026-08-12", "Revisiting quantization: FP8 on new GPUs", [
        "Is FP8 inference worth it compared to int8 or AWQ 4-bit?",
        "On GPUs with FP8 Tensor Cores (Hopper, Ada, Blackwell), FP8 weights and activations give close to fp16 quality with about twice the throughput, and vLLM supports FP8 checkpoints and FP8 KV cache. AWQ 4-bit still wins on memory, so on a 12 GB consumer GPU it remains the practical option for 7B models. The choice is memory-bound versus compute-bound: quantization accelerates decoding mostly by reducing memory traffic.",
    ]),
    ("claude", "2026-09-15", "Temporal knowledge graphs: tracking how ideas evolve", [
        "How could my knowledge graph track how my understanding of a topic changes over time?",
        "Store timestamps on mentions and edges: first_seen, last_seen and per-month counts. Then you can show a timeline per concept, detect topics you have not revisited in months, and compare communities across periods to see which interests grew. Temporal edges also help retrieval for questions like 'what did I decide about vLLM last spring'.",
        "Could it find connections I never made explicitly?",
        "Link prediction does exactly that: pairs of concepts with many shared neighbors (Adamic–Adar or Jaccard) but no direct edge are candidates for an idea you have not connected yet. For example quantization might link your TinyML work on the ESP32 and your LLM serving experiments even if you never discussed them together.",
    ]),
]
