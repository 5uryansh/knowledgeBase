# Getting your chat exports

knowledgeBase reads the official export of each assistant. Put the files where `kb build` looks by default,
or point to them with `--export <platform> <path>` / the `exports:` section of `kb.yaml`.

| platform | where it goes (default) |
|---|---|
| ChatGPT | `data/chatgpt/conversations.json` |
| Claude | `data/claude/conversations.json` |
| Gemini | `data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json` |
| Claude Code | read from `~/.claude/projects` automatically (or copy it to `data/claude-code/projects`) |

## ChatGPT

1. chatgpt.com → your avatar → **Settings → Data controls → Export data → Export**.
2. You get an email with a download link (usually within minutes). Unzip it.
3. Copy `conversations.json` to `data/chatgpt/`.

Only the active branch of edited conversations is imported, so regenerated answers don't duplicate content.

## Claude (claude.ai)

1. claude.ai → **Settings → Privacy → Export data**.
2. Download the archive from the email and unzip it.
3. Copy `conversations.json` to `data/claude/`.

Extended-thinking blocks are skipped by default (they are long and repeat the answer).

## Gemini

Gemini conversations are part of Google Takeout's *My Activity*:

1. Go to takeout.google.com → **Deselect all** → select **My Activity**.
2. Under *My Activity* → **All activity data included** → keep only **Gemini Apps**; set the format to **JSON**.
3. Export, download, unzip, and copy the `Takeout` folder into `data/gemini/` so the path is
   `data/gemini/Takeout/My Activity/Gemini Apps/MyActivity.json`.

Takeout has no conversation ids, so activity from the same day becomes one conversation.

## Claude Code

Claude Code keeps every session as a JSONL file under `~/.claude/projects/<project>/`. `kb build` reads that
folder directly when `data/claude-code/projects` doesn't exist. Sub-agent ("sidechain") turns and slash-command
noise are skipped. On WSL, the folder is the one inside the Linux home where you run `claude`.

## Re-importing

Export again whenever you like and re-run `kb build`. Conversations are identified by their platform id, so
re-imports update instead of duplicating, deleted conversations disappear from the vault, and concept
extraction is cached — only new or changed conversations are processed.

## Privacy

Everything stays on your machine. Nothing is uploaded unless you configure an LLM for `kb summarize`
(then only the excerpts for the summarised concepts are sent to that endpoint) or publish a static export
(conversation text is only included with `--include-conversations`).
