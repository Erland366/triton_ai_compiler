"""Correctness-first agent for AMD kernels, adapted from the Triton AI Compiler's loop.

Drives Claude Code through the Claude Agent SDK, logged in with the user's own
subscription (docs/adr/0002-restricted-claude-code-agent.md). The agent has no
built-in tools: only read, patch, verify, and reset on one candidate file. Every
verified candidate and report is kept in the run directory.

    python -m taic_amd.agent --kernel MatrixMultiplicationBF16OutputFloat32 \
        --capture captures/a100-ref-1 --hours 12 --out runs
    python -m taic_amd.agent ... --dry-run     # check the tools without a model

Run agents only inside the agent container, after the user has approved the run.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    RateLimitEvent,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from bf16_kernels import KERNELS
from taic_amd.prompts import SYSTEM_PROMPT, task_prompt
from taic_amd.verifier import summary, verify

SERVER_NAME = "taic"
MAX_LINES_PER_READ = 400
DEFAULT_LIMIT_WAIT = 30 * 60


def detail(report):
    lines = [summary(report)]
    if report["status"] == "ok":
        for result in report["visible"]["per_input"]:
            lines.append(
                f"  {result['name']}: "
                f"{result['bit_equal'] / result['values']:.4%} bit-equal, "
                f"{result['within_1_step'] / result['values']:.4%} within 1 step, "
                f"max {result['max_steps']} steps"
                + ("" if result["nan_positions_agree"] else ", NaN positions differ")
            )
    return "\n".join(lines)


def rank(report):
    if report["status"] != "ok":
        return (0, 0.0, 0.0, 0.0)
    exact = report["exact"]
    return (
        int(exact),
        -report["latency_ms"] if exact else 0.0,
        report["visible"]["bit_equal_fraction"],
        report["visible"]["within_1_step_fraction"],
    )


class Session:
    def __init__(self, run_dir, kernel, capture, stage, start_text, timeout):
        self.run_dir, self.kernel, self.capture = run_dir, kernel, capture
        self.stage, self.timeout = stage, timeout
        self.candidate_name = f"candidate.{stage}"
        self.current = start_text
        self.verified = {}
        self.best_id = None
        (run_dir / "candidates").mkdir(parents=True)
        self.trace = (run_dir / "trace.jsonl").open("a")

    def log(self, event, **fields):
        record = {"time": datetime.now(UTC).isoformat(), "event": event, **fields}
        self.trace.write(json.dumps(record) + "\n")
        self.trace.flush()

    def read(self, start_line, end_line):
        lines = self.current.splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line, start + MAX_LINES_PER_READ - 1)
        shown = "\n".join(f"{n:6d}  {lines[n - 1]}" for n in range(start, end + 1))
        return f"{self.candidate_name}: lines {start}-{end} of {len(lines)}\n{shown}"

    def apply_patch(self, patch):
        headers = [
            line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))
        ]
        expected = [f"--- {self.candidate_name}", f"+++ {self.candidate_name}"]
        if [h.split("\t")[0] for h in headers] != expected:
            raise ValueError(
                f"Patch headers must be exactly {expected}, got {headers}."
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.candidate_name
            path.write_text(self.current)
            result = subprocess.run(
                ["patch", "--batch", "--forward", "--strip=0", "--input=-"],
                cwd=directory, input=patch, text=True, capture_output=True, check=False,
            )  # fmt: skip
            if result.returncode != 0:
                raise ValueError(
                    f"Patch did not apply: {(result.stdout + result.stderr).strip()}"
                )
            self.current = path.read_text()
        self.log("patch", lines=len(self.current.splitlines()))
        return f"Applied. {self.candidate_name} now has {len(self.current.splitlines())} lines."

    def verify(self):
        candidate_id = len(self.verified)
        path = self.run_dir / "candidates" / f"{candidate_id:03d}.{self.stage}"
        path.write_text(self.current)
        report = verify(self.kernel, self.capture, candidate=path, timeout=self.timeout)
        (path.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
        self.verified[candidate_id] = (self.current, report)
        if self.best_id is None or rank(report) > rank(self.verified[self.best_id][1]):
            self.best_id = candidate_id
        self.log("verify", candidate=candidate_id, summary=summary(report))
        return f"Candidate {candidate_id}: {detail(report)}\nBest so far: candidate {self.best_id}."

    def reset(self, candidate_id):
        if candidate_id not in self.verified:
            raise ValueError(
                f"No verified candidate {candidate_id}; have {sorted(self.verified)}."
            )
        self.current = self.verified[candidate_id][0]
        self.log("reset", candidate=candidate_id)
        return f"Current candidate is now a copy of candidate {candidate_id}."


def text_result(function):
    async def handler(args):
        try:
            return {
                "content": [
                    {"type": "text", "text": await asyncio.to_thread(function, args)}
                ]
            }
        except ValueError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "is_error": True}

    return handler


def build_server(session):
    tools = [
        tool("read_candidate", "Show numbered lines of the current candidate.",
             {"start_line": int, "end_line": int})(
            text_result(lambda a: session.read(a["start_line"], a["end_line"]))),
        tool("apply_patch", "Apply a unified diff to the current candidate.", {"patch": str})(
            text_result(lambda a: session.apply_patch(a["patch"]))),
        tool("verify", "Compile, run, and compare the current candidate with A100.", {})(
            text_result(lambda a: session.verify())),
        tool("reset", "Make an earlier verified candidate current again.", {"candidate_id": int})(
            text_result(lambda a: session.reset(a["candidate_id"]))),
    ]  # fmt: skip
    return create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0", tools=tools)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", default="llir", choices=("llir", "amdgcn"))
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-sessions", type=int, default=20)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--verify-timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class Clock:
    """Active time: wall time minus the time spent waiting out usage limits."""

    def __init__(self, hours):
        self.budget = hours * 3600
        self.started = time.monotonic()
        self.paused = 0.0

    def active(self):
        return time.monotonic() - self.started - self.paused

    def remaining(self):
        return self.budget - self.active()

    async def pause_until(self, resets_at):
        began = time.monotonic()
        await asyncio.sleep(max(60.0, resets_at - time.time() + 60.0))
        self.paused += time.monotonic() - began


def agent_options(session, args, resume):
    env = {}
    if os.environ.get("TAIC_CLAUDE_CONFIG_DIR"):
        env["CLAUDE_CONFIG_DIR"] = os.environ["TAIC_CLAUDE_CONFIG_DIR"]
    return ClaudeAgentOptions(
        tools=[],
        mcp_servers={SERVER_NAME: build_server(session)},
        allowed_tools=[f"mcp__{SERVER_NAME}__*"],
        permission_mode="dontAsk",
        setting_sources=[],
        system_prompt=SYSTEM_PROMPT,
        model=args.model,
        effort=args.effort,
        max_turns=args.max_turns,
        cwd=str(session.run_dir),
        env=env,
        resume=resume,
    )


async def run_one_session(session, options, prompt, clock):
    """Run the agent until it ends its turn, hits a usage limit, or runs out of time.

    Returns (outcome, session_id, resets_at), outcome being "ended", "rate_limited",
    "time", or "error".
    """
    outcome, session_id, resets_at = "ended", options.resume, None
    async with ClaudeSDKClient(options=options) as client:

        async def stop_when_out_of_time():
            nonlocal outcome
            while clock.remaining() > 0:
                await asyncio.sleep(min(30.0, max(1.0, clock.remaining())))
            outcome = "time"
            session.log("time_limit_reached", active_seconds=clock.active())
            await client.interrupt()

        watchdog = asyncio.create_task(stop_when_out_of_time())
        try:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    session.log("init", tools=message.data.get("tools"),
                                model=message.data.get("model"))  # fmt: skip
                elif isinstance(message, RateLimitEvent):
                    info = message.rate_limit_info
                    session.log("rate_limit", status=info.status, resets_at=info.resets_at,
                                window=info.rate_limit_type, utilization=info.utilization)  # fmt: skip
                    if info.status == "rejected":
                        outcome, resets_at = "rate_limited", info.resets_at
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            session.log("text", text=block.text[:4000])
                        elif isinstance(block, ToolUseBlock):
                            session.log("tool_use", name=block.name,
                                        input=json.dumps(block.input)[:2000])  # fmt: skip
                    if message.error == "rate_limit" and outcome != "time":
                        outcome = "rate_limited"
                    elif message.error:
                        session.log("assistant_error", error=message.error)
                elif isinstance(message, ResultMessage):
                    session_id = message.session_id
                    session.log(
                        "result", subtype=message.subtype, is_error=message.is_error,
                        num_turns=message.num_turns, duration_ms=message.duration_ms,
                        duration_api_ms=message.duration_api_ms, session_id=message.session_id,
                    )  # fmt: skip
                    if message.is_error and outcome == "ended":
                        outcome = "error"
        finally:
            watchdog.cancel()
    return outcome, session_id, resets_at


async def run_agent(session, args, native_summary, clock):
    prompt = task_prompt(
        KERNELS[args.kernel], session.config, args.stage, session.candidate_name,
        native_summary, len(session.current.splitlines()),
    )  # fmt: skip
    resume = None
    for number in range(args.max_sessions):
        if clock.remaining() <= 0:
            break
        session.log("session_start", number=number, resume=resume,
                    active_seconds=clock.active())  # fmt: skip
        options = agent_options(session, args, resume)
        outcome, resume, resets_at = await run_one_session(
            session, options, prompt, clock
        )
        session.log("session_end", number=number, outcome=outcome)
        if outcome in ("time", "error"):
            return outcome
        if outcome == "rate_limited":
            wait_until = resets_at or time.time() + DEFAULT_LIMIT_WAIT
            session.log("pause", until=wait_until)
            await clock.pause_until(wait_until)
            prompt = "The usage limit has reset. Continue from where you stopped."
            continue
        best = session.verified.get(session.best_id)
        prompt = (
            "You ended your turn, but time remains. The best candidate so far is "
            f"{session.best_id}: {summary(best[1]) if best else 'none verified yet'}. "
            "If no candidate is exact yet, keep working toward an exact match. If one "
            "is exact, make it faster while keeping it exact."
        )
    return "sessions_exhausted"


def main():
    args = parse_args()
    capture = args.capture.resolve()
    manifest = json.loads((capture / "manifest.json").read_text())
    config = manifest["kernels"][args.kernel]["configs"]["tuned"]["config"]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.out / f"{stamp}_{args.kernel}").resolve()
    native_dir = run_dir / "native"
    native = verify(
        args.kernel, capture, dump_native=native_dir, timeout=args.verify_timeout
    )
    if native["status"] != "ok":
        raise RuntimeError(f"Native kernel failed to verify: {summary(native)}")
    start_text = (native_dir / f"kernel.{args.stage}").read_text()
    session = Session(
        run_dir, args.kernel, capture, args.stage, start_text, args.verify_timeout
    )
    session.config = config
    session.log("start", sdk=claude_agent_sdk.__version__, kernel=args.kernel, capture=manifest["name"], stage=args.stage,
                model=args.model, effort=args.effort, hours=args.hours,
                dry_run=args.dry_run, native=summary(native))  # fmt: skip
    clock = Clock(args.hours)
    stopped = "dry_run"
    if args.dry_run:
        print(session.read(1, 5))
        print(session.verify())
    else:
        stopped = asyncio.run(run_agent(session, args, detail(native), clock))
    best = session.verified.get(session.best_id)
    result = {
        "run_dir": str(run_dir),
        "stopped": stopped,
        "active_seconds": clock.active(),
        "paused_seconds": clock.paused,
        "wall_seconds": clock.active() + clock.paused,
        "verified_candidates": len(session.verified),
        "best_candidate": session.best_id,
        "best_summary": summary(best[1]) if best else None,
        "native_summary": summary(native),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
