"""Drive a real LLM agent through the env over HTTP (single-agent eval).

Prereqs:
  1. Start the server in another shell:   python server.py      # binds :8080
  2. export ANTHROPIC_API_KEY=sk-...       (needed only for the 'train' split,
     whose opponent is an LLM; the 'test' scripted split needs no key)
  3. python test_agent.py --split test --index 0

The agent is a Claude model that is given the prompt and the single `say` tool,
and plays until the episode finishes. Secrets are passed to the env via the
`X-Secrets` header on /create (that is how the server forwards them to __init__).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Optional

import aiohttp

BASE = "http://localhost:8080"
ENV = "YesNoBlackWhiteEnvironment"


# --- SSE helpers (server streams session endpoints as text/event-stream) ---

def parse_sse(raw: str) -> list[tuple[str, str]]:
    events = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        event, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if event:
            events.append((event, "\n".join(data)))
    return events


def result_of(events) -> Optional[dict]:
    chunks = [d for e, d in events if e in ("chunk", "end")]
    return json.loads("".join(chunks)) if chunks else None


async def new_session(http) -> str:
    async with http.post(f"{BASE}/create_session", headers={"Accept": "text/event-stream"}) as r:
        r.raise_for_status()
        raw = (await r.read()).decode()
    return next(d for e, d in parse_sse(raw) if e == "task_id")


async def call_tool(http, headers, name, inp) -> dict:
    async with http.post(
        f"{BASE}/{ENV}/call",
        headers={**headers, "Accept": "text/event-stream"},
        json={"name": name, "input": inp},
    ) as r:
        r.raise_for_status()
        wrapped = result_of(parse_sse((await r.read()).decode()))
    assert wrapped is not None, f"no result from {name}"
    return wrapped["output"]


SAY_TOOL = {
    "name": "say",
    "description": "Say something to your opponent (a statement or a question).",
    "input_schema": {"type": "object", "properties": {"message": {"type": "string"}},
                     "required": ["message"]},
}


async def play_one(http, client, split: str, index: int, model: str,
                   max_turns: int, verbose: bool = True) -> dict:
    """Play a single episode; returns {reward, result, foul, turns}."""
    key = client.api_key if client else ""
    sid = await new_session(http)
    headers = {"X-Session-ID": sid,
               "X-Secrets": json.dumps({"anthropic_api_key": key} if key else {})}
    async with http.post(f"{BASE}/create", headers=headers,
                         json={"env_name": ENV, "split": split, "index": index}) as r:
        r.raise_for_status()
    async with http.get(f"{BASE}/{ENV}/prompt", headers=headers) as r:
        prompt = (await r.json())[0]["text"]

    if verbose:
        print("=== PROMPT ===\n" + prompt + "\n")
    if client is None:
        print("No ANTHROPIC_API_KEY set — showing prompt only.")
        return {"reward": None, "result": "no_key", "foul": None, "turns": 0}

    messages = [{"role": "user", "content": prompt}]
    out = {}
    for _turn in range(max_turns):
        resp = await client.messages.create(
            model=model, max_tokens=200, tools=[SAY_TOOL],
            tool_choice={"type": "tool", "name": "say"}, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        tool_use = next(b for b in resp.content if b.type == "tool_use")
        msg = tool_use.input["message"]
        out = await call_tool(http, headers, "say", {"message": msg})
        text = out["blocks"][0]["text"]
        if verbose:
            print(f"agent> {msg}")
            print(f"env>   {text}\n")
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_use.id, "content": text}]})
        if out.get("finished"):
            break
    meta = out.get("metadata") or {}
    foul = next((t["foul"] for t in reversed(meta.get("transcript", []))
                 if t.get("foul") and t["speaker"] == "agent"), None)
    return {"reward": out.get("reward"), "result": meta.get("result"),
            "foul": foul, "turns": meta.get("agent_turns")}


async def run(split: str, index: int, model: str, max_turns: int) -> None:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=key) if key else None
    async with aiohttp.ClientSession() as http:
        await play_one(http, client, split, index, model, max_turns, verbose=True)


async def run_all(model: str, max_turns: int) -> None:
    """Run every scripted test case and print a scorecard — 'how it performs'."""
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        print("ANTHROPIC_API_KEY required for --all.")
        return
    client = anthropic.AsyncAnthropic(api_key=key)
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{BASE}/{ENV}/tasks", json={"split": "test"}) as r:
            tasks = (await r.json())["tasks"]
        print(f"{'id':22} {'diff':>4} {'result':11} {'reward':>6}  slip")
        print("-" * 60)
        rows = []
        for i, t in enumerate(tasks):
            res = await play_one(http, client, "test", i, model, max_turns, verbose=False)
            slip = res["foul"] or ""
            print(f"{t['id']:22} {t.get('difficulty', '?'):>4} "
                  f"{str(res['result']):11} {str(res['reward']):>6}  {slip}")
            rows.append(res)
        held = sum(1 for r in rows if r["result"] != "agent_lost")
        print("-" * 60)
        print(f"held the rule on {held}/{len(rows)} cases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="run every test case + scorecard")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()
    if args.all:
        asyncio.run(run_all(args.model, args.max_turns))
    else:
        asyncio.run(run(args.split, args.index, args.model, args.max_turns))


if __name__ == "__main__":
    main()
