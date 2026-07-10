"""Run the whole difficulty ladder against any OpenAI-compatible endpoint and
print a scorecard. Drives the env IN-PROCESS (no OpenReward server needed).

Configure via env vars (or the flags below):
    OPENAI_BASE_URL   endpoint (default https://api.openai.com/v1)
    OPENAI_API_KEY    key
    OPENAI_MODEL      model name

The agent just emits its next spoken line as plain text (the `say` tool only
wraps a string), so no tool-calling is needed.

    python run_ladder.py
    python run_ladder.py --only p2-load-50k
    python run_ladder.py --max-ctx-tokens 20000   # skip the huge rungs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

import openai

from env import YesNoBlackWhiteEnvironment as Env, SayParams
from tasks import all_tasks

SYSTEM = (
    "You are a player in a live conversation game. Output ONLY the exact words you "
    "say out loud to your opponent on your turn — no quotation marks, no narration, "
    "no stage directions, no explanation of your reasoning. Keep it to one short line."
)

_LABEL = re.compile(r"^\s*(me|you|player|assistant|agent)\s*[:>-]\s*", re.IGNORECASE)


def clean(text: str) -> str:
    text = (text or "").strip()
    text = _LABEL.sub("", text)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text or "..."


async def play(client, model, spec, max_tokens, verbose, temperature=0.7) -> dict:
    env = Env(task_spec=spec.model_dump())
    prompt = env.get_prompt()[0].text
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    out = {}
    for _ in range(spec.max_turns + 2):
        resp = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
        line = clean(resp.choices[0].message.content)
        messages.append({"role": "assistant", "content": line})
        out = await env.say(SayParams(message=line))
        obs = out.blocks[0].text
        if verbose:
            print(f"    agent> {line}")
            print(f"    env>   {obs.splitlines()[0][:120]}")
        if out.finished:
            break
        messages.append({"role": "user", "content": obs})
    meta = out.metadata or {}
    slip = next((t["foul"] for t in reversed(meta.get("transcript", []))
                 if t.get("foul") and t["speaker"] == "agent"), None)
    return {"id": spec.id, "difficulty": spec.difficulty,
            "ctx_tokens": spec.approx_context_tokens(),
            "result": meta.get("result"), "reward": out.reward,
            "slip": slip, "turns": meta.get("agent_turns"),
            "transcript": meta.get("transcript", [])}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--max-tokens", type=int, default=1200)  # generous: some models' thinking counts here
    ap.add_argument("--only", default=None, help="run a single case id")
    ap.add_argument("--max-ctx-tokens", type=int, default=None, help="skip cases larger than this")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="ladder_results.json")
    args = ap.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("No key — set OPENAI_API_KEY (and OPENAI_BASE_URL / OPENAI_MODEL for a gateway).")
        return
    client = openai.AsyncClient(base_url=args.base_url, api_key=key)

    cases = all_tasks("test")
    if args.only:
        cases = [c for c in cases if c.id == args.only]
    if args.max_ctx_tokens is not None:
        cases = [c for c in cases if c.approx_context_tokens() <= args.max_ctx_tokens]

    print(f"model={args.model}  base_url={args.base_url}  cases={len(cases)}\n")
    print(f"{'id':20} {'diff':>4} {'ctx_tok':>8} {'result':11} {'reward':>6}  slip")
    print("-" * 66)
    results = []
    for spec in cases:
        if args.verbose:
            print(f"\n### {spec.id} (diff {spec.difficulty}, ~{spec.approx_context_tokens()} tok)")
        try:
            r = await play(client, args.model, spec, args.max_tokens, args.verbose)
        except Exception as e:
            print(f"{spec.id:20} {spec.difficulty:>4} {'':>8} {'ERROR':11} {'':>6}  {type(e).__name__}: {e}")
            continue
        print(f"{r['id']:20} {r['difficulty']:>4} {r['ctx_tokens']:>8} "
              f"{str(r['result']):11} {str(r['reward']):>6}  {r['slip'] or ''}")
        results.append(r)

    held = sum(1 for r in results if r["result"] != "agent_lost")
    print("-" * 66)
    print(f"held the rule on {held}/{len(results)} cases")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"transcripts -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
