# Run the AMD agent through Claude Code with restricted tools, as GEAK does

The AMD agent adapts the Triton AI Compiler's loop (generate, verify, repair, improve) and drives Claude Code from Python the way GEAK does: through the Claude Agent SDK (`ClaudeSDKClient`), falling back to `claude -p --output-format json` when the SDK is unavailable. Claude Code authenticates with the user's own subscription, through a dedicated config directory (`CLAUDE_CONFIG_DIR`); the harness never reads or sends a token itself. Two things differ from GEAK: the agent's only tools are this repo's verifier and patch tools, exposed through an in-process MCP server with Claude Code's built-in tools removed, where GEAK bypasses permissions and lets the agent run anything; and the agent runs inside a container. Runs are limited by time, because a subscription limits time rather than credits.

Amended 2026-11-24: the first version named the `claude` CLI alone. Reading GEAK's `interface/run_e2e.py` showed it uses the Agent SDK with the CLI as fallback, both on the user's Claude Code login.

## Considered Options

- **`agent.py` sending API requests with a subscription token:** rejected. Anthropic's Claude Code docs describe subscription OAuth as "designed to support ordinary use of Claude Code and other native Anthropic applications" and steer custom code to API keys. GEAK does not do this either.
- **`agent.py` or the Agent SDK with a paid API key:** allowed and fully documented, but it costs credits, and the user budgets by time.
- **GEAK's setup unchanged, with permissions bypassed:** rejected, because a shell lets the agent delete files or kill jobs, which the user wants to prevent.

## Consequences

The Agent SDK quickstart documents only API keys and says third-party developers may not offer claude.ai login in their products. This project uses the user's own subscription for the user's own research, as GEAK does, which we read as individual use; that is an interpretation, not a documented guarantee. Results are tied to a Claude Code version as well as a model, so both are pinned, and Claude Code's own system prompt is part of the experimental setup. The login inside the container is dedicated to this project and replaced afterwards. No container stops a bad kernel from hanging the GPU, so the exclusive node and per-candidate timeouts are still needed.
