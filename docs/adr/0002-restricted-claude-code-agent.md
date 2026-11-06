# Run the AMD agent as Claude Code with restricted tools

The AMD agent runs through the `claude` CLI with a subscription token rather than through `agent.py`, limited to evaluate and patch tools by `--tools`, `--strict-mcp-config`, and an MCP server, and inside a container. Runs are limited by time because a subscription limits time rather than credits, and comparison with the Triton AI Compiler paper is not a priority, so keeping its exact agent design is worth little.

## Considered Options

- **`agent.py` with a subscription token:** rejected. It would make direct API calls through the Anthropic SDK, and press reports say Anthropic bars subscription tokens outside Claude Code and claude.ai. The signals conflict and the current terms were not checked, so this route was not built on.
- **`agent.py` with a paid API key:** allowed, but it costs credits, and the user budgets by time.
- **Claude Code with a full shell, as GEAK does:** rejected because a shell lets the agent delete files or kill jobs, which is what the user wants to prevent. GEAK's README and setup guide describe no container, and run it with permissions auto-approved. A later comparison remains possible.

## Consequences

Results are tied to a Claude Code version as well as a model, so both are pinned. Claude Code's own system prompt becomes part of the experimental setup. The token sits inside the container, so it is dedicated to this project and replaced afterwards. No container stops a bad kernel from hanging the GPU, so the exclusive node and per-candidate timeouts are still needed.
