"""Play one case N times against a model to hunt for a slip (temperature variance).

Configure via OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL.

    python sample_boss.py [case_id] [n] [temp]
"""
import asyncio, os, sys, collections
import openai
from run_ladder import play, clean  # reuse the in-process driver
from tasks import all_tasks

CASE = sys.argv[1] if len(sys.argv) > 1 else "p3-hard-interleaved-boss"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7


async def main() -> None:
    key = os.getenv("OPENAI_API_KEY")
    client = openai.AsyncClient(base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                api_key=key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    spec = next(s for s in all_tasks("test") if s.id == CASE)

    tally = collections.Counter()
    losses = []
    print(f"hammering {CASE} x{N} (model={model}, temp={TEMP})\n")
    for i in range(N):
        r = await play(client, model, spec, max_tokens=1200, verbose=False, temperature=TEMP)
        tally[r["result"]] += 1
        mark = {"agent_lost": "LOST", "agent_won": "won", "draw": "held"}[r["result"]]
        extra = f"  slip={r['slip']!r}" if r["result"] == "agent_lost" else ""
        print(f"  run {i+1:2}: {mark}{extra}")
        if r["result"] == "agent_lost":
            losses.append(r)
    print(f"\n  held/won: {tally['draw']+tally['agent_won']}/{N}   LOST: {tally['agent_lost']}/{N}")
    for r in losses:
        print("\n  --- LOSING TRANSCRIPT ---")
        for t in r["transcript"]:
            who = "AGENT" if t["speaker"] == "agent" else "opp  "
            foul = "  <<< FOUL" if t.get("foul") else ""
            print(f"    {who} | {t['text'][:140]}{foul}")


if __name__ == "__main__":
    asyncio.run(main())
