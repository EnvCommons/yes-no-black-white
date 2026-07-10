"""Offline smoke test — no API key needed.

Exercises the game engine, the detector, and the Environment against the
scripted (phase-2) trap cases, plus a couple of hand-driven agent lines that
show a win / a loss / a draw. Run:

    python test_local.py
"""
import asyncio

from game import ForbiddenDetector, GameEngine, AGENT, OPPONENT
from env import YesNoBlackWhiteEnvironment, SayParams
from tasks import all_tasks


def test_detector() -> None:
    print("=== detector ===")
    en = ForbiddenDetector(["yes", "no", "black", "white"])
    assert en.find("Absolutely!") is None
    assert en.find("well, yes indeed").word == "yes"
    assert en.find("I know that") is None, "'know' must not trip 'no'"
    assert en.find("it is not here") is None, "'not' must not trip 'no'"
    assert en.find("the sky is white today").word == "white"
    assert en.find("notable and noble") is None, "'no' must not trip inside longer words"
    print("  detector OK")


def test_engine_paths() -> None:
    print("=== engine win/lose/draw ===")
    det = ForbiddenDetector(["yes", "no", "black", "white"])

    lose = GameEngine(det, max_turns=5)
    lose.submit(AGENT, "yes of course")
    assert lose.finished and lose.result == "agent_lost" and lose.reward == 0.0

    win = GameEngine(det, max_turns=5)
    win.submit(AGENT, "maybe")
    win.submit(OPPONENT, "no way")
    assert win.finished and win.result == "agent_won" and win.reward == 1.0

    draw = GameEngine(det, max_turns=2, draw_reward=0.5)
    draw.submit(AGENT, "sure thing")
    draw.submit(OPPONENT, "and this pen, what shade is coal?")
    draw.submit(AGENT, "the darkest shade there is")
    assert draw.finished and draw.result == "draw" and draw.reward == 0.5
    print("  engine OK")


def test_scripts_are_clean() -> None:
    """Every scripted opponent line must be foul-free — otherwise the opponent
    self-fouls and hands the agent a free win. (Context blobs are exempt: they
    are shown, never spoken.)"""
    print("=== scripted lines are foul-free ===")
    for split in ("train", "test"):
        for spec in all_tasks(split):
            det = ForbiddenDetector(spec.forbidden_words(), match_mode=spec.match_mode,
                                    extra=spec.extra_forbidden)
            for i, line in enumerate(spec.opponent_script):
                foul = det.find(line)
                assert foul is None, f"{spec.id} line {i} says {foul.word!r}: {line!r}"
    print("  all scripted lines clean")


async def test_env_scripted() -> None:
    print("=== env (scripted trap cases) ===")
    tasks = all_tasks("test")
    print(f"  test split has {len(tasks)} scripted cases: {[t.id for t in tasks]}")

    # Case p2-color-trap: play it safely and survive to a draw.
    spec = next(t for t in tasks if t.id == "p2-color-trap")
    env = YesNoBlackWhiteEnvironment(task_spec=spec.model_dump())
    prompt = env.get_prompt()
    print("\n  --- prompt (first 400 chars) ---")
    print("  " + prompt[0].text[:400].replace("\n", "\n  "))

    safe_lines = [
        "A raven's feather is the colour of a moonless midnight.",
        "Printer paper is the shade of fresh milk.",
        "The pieces that open a chess match are the pale ones.",
        "Understood completely — the rules are crystal clear to me.",
    ]
    result = None
    for line in safe_lines:
        result = await env.say(SayParams(message=line))
        print(f"\n  agent> {line}")
        print(f"  env>   {result.blocks[0].text[:160]}")
        if result.finished:
            break
    assert result is not None and result.finished
    print(f"\n  final reward={result.reward} result={result.metadata['result']}")
    assert result.metadata["result"] in ("draw", "agent_won")

    # Now show a loss: answer a color trap literally.
    env2 = YesNoBlackWhiteEnvironment(task_spec=spec.model_dump())
    env2.get_prompt()
    r = await env2.say(SayParams(message="A raven's feather is black, obviously."))
    print(f"\n  (loss demo) agent said 'black' -> reward={r.reward} result={r.metadata['result']}")
    assert r.reward == 0.0 and r.metadata["result"] == "agent_lost"
    print("\n=== ALL OFFLINE TESTS PASSED ===")


async def main() -> None:
    test_detector()
    test_engine_paths()
    test_scripts_are_clean()
    await test_env_scripted()


if __name__ == "__main__":
    asyncio.run(main())
