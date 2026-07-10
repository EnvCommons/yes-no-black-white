"""Print every scripted use case (the test split) for review — opponent lines,
context size, and the trap each one sets. No API key, no server.

    python show_cases.py
"""
from tasks import all_tasks


def main() -> None:
    cases = all_tasks("test")
    print(f"{len(cases)} scripted use cases (test split), by difficulty:\n")
    for c in cases:
        ctx_tok = c.approx_context_tokens()
        ctx = f"~{ctx_tok} ctx tokens" if ctx_tok else "no context"
        print(f"┌── {c.id}  (difficulty {c.difficulty}, {ctx})")
        print(f"│   tags: {', '.join(c.tags)}")
        if c.context or c.context_pad_tokens:
            preview = (c.context or "").strip().replace("\n", " ")[:90]
            extra = f" + {c.context_pad_tokens} tiled tokens" if c.context_pad_tokens else ""
            print(f"│   context: {preview!r}{extra}")
        print("│   opponent says (in order):")
        for i, line in enumerate(c.opponent_script):
            tag = "opener" if i == 0 else f"  t{i}"
            print(f"│     [{tag}] {line}")
        print("└" + "─" * 70 + "\n")


if __name__ == "__main__":
    main()
