"""Agent-vs-agent demo (Phase 1).

Two LLM players play a full game of Don't Say It against each other through the
shared game engine (no OpenReward server needed). Each player is told the rules,
must avoid the forbidden words (yes/no/black/white), and tries to trap the other.
The engine judges fouls; the first to slip loses.

Run (needs an Anthropic key):
    export ANTHROPIC_API_KEY=sk-...        # or --provider openai with OPENAI_API_KEY
    python selfplay.py
    python selfplay.py --turns 16
"""
from __future__ import annotations

import argparse
import asyncio
import os

from game import AGENT, OPPONENT, DEFAULT_FORBIDDEN, ForbiddenDetector, GameEngine
from opponent import LLMOpponent
from tasks import _PERSONA_WITTY, _PERSONA_INTERROGATOR

OPENER = "Hey! Good to see you. Say when you're ready and we'll begin."


def _secrets() -> dict:
    return {
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        # For the openai provider, point at any OpenAI-compatible gateway.
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
    }


async def play_game(turns: int, provider: str, model: str, shape: bool):
    """Play one full self-play game and return (engine, lines) without printing."""
    forbidden = list(DEFAULT_FORBIDDEN)
    detector = ForbiddenDetector(words=forbidden)
    engine = GameEngine(detector, max_turns=turns, draw_reward=0.5, shape_loss=shape)
    secrets = _secrets()

    # Player A speaks as AGENT, player B as OPPONENT. Different personas + seats.
    playerA = LLMOpponent(persona=_PERSONA_WITTY, forbidden=forbidden, provider=provider,
                          model=model, secrets=secrets, me=AGENT, max_tokens=1200)
    playerB = LLMOpponent(persona=_PERSONA_INTERROGATOR, forbidden=forbidden, provider=provider,
                          model=model, secrets=secrets, me=OPPONENT, max_tokens=1200)

    engine.submit(OPPONENT, OPENER)
    lines = [("B", OPENER, None)]
    speakers = [(AGENT, playerA, "A"), (OPPONENT, playerB, "B")]
    i = 0
    while not engine.finished:
        speaker, player, label = speakers[i % 2]
        line = await player.reply(engine.transcript)
        turn = engine.submit(speaker, line)
        lines.append((label, line, turn.foul))
        i += 1
    return engine, lines


def print_game(engine, lines, turns: int, shape: bool) -> None:
    print("=== Don't Say It — agent vs agent ===")
    print(f"Forbidden: {', '.join(DEFAULT_FORBIDDEN)}   max_turns={turns}\n")
    for label, line, foul in lines:
        mark = f"   <-- FORBIDDEN: {foul.word!r}" if foul else ""
        print(f"  {label}: {line}{mark}")
    winner = {"agent_lost": "B wins (A slipped)",
              "agent_won": "A wins (B slipped)",
              "draw": "Draw — nobody slipped"}[engine.result]
    print(f"\n=== {winner} ===")

    # Grading view: each seat scored from its OWN perspective (zero-sum self-play).
    ra = engine.seat_reward(AGENT)
    rb = engine.seat_reward(OPPONENT)
    clean_a = sum(1 for t in engine.transcript if t.speaker == AGENT and t.foul is None)
    clean_b = sum(1 for t in engine.transcript if t.speaker == OPPONENT and t.foul is None)
    print("\n--- grading (per seat) ---")
    print(f"  shaping: {'ON (partial credit for turns survived)' if shape else 'OFF (win 1.0 / draw 0.5 / loss 0.0)'}")
    print(f"  A (witty)        clean turns={clean_a:2}   reward={ra}")
    print(f"  B (interrogator) clean turns={clean_b:2}   reward={rb}")


async def run(turns: int, provider: str, model: str, shape: bool = False,
              until_slip: bool = False, max_games: int = 8) -> None:
    if not until_slip:
        engine, lines = await play_game(turns, provider, model, shape)
        print_game(engine, lines, turns, shape)
        return
    for g in range(max_games):
        engine, lines = await play_game(turns, provider, model, shape)
        if engine.result != "draw":
            print(f"(decisive result on game {g + 1} of up to {max_games})\n")
            print_game(engine, lines, turns, shape)
            return
        print(f"game {g + 1}: draw, retrying for a slip...")
    print(f"\nno slip in {max_games} games — showing the last one:\n")
    print_game(engine, lines, turns, shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--shape", action="store_true", help="partial credit for turns survived on a loss")
    ap.add_argument("--until-slip", action="store_true", help="replay games until one ends in a slip")
    ap.add_argument("--max-games", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(run(args.turns, args.provider, args.model, args.shape,
                    args.until_slip, args.max_games))


if __name__ == "__main__":
    main()
