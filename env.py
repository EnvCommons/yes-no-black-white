"""OpenReward environment: "Yes, No, Black, White" (the forbidden-words game).

The agent plays one side of the forbidden-words game against an opponent (a
scripted trap or an LLM). The agent speaks via the `say` tool. Whoever says a
forbidden word (yes / no / black / white) first loses:

  * agent says a forbidden word    -> reward 0.0  (finished)
  * opponent says a forbidden word -> reward 1.0  (finished)
  * agent survives max_turns        -> reward draw_reward (default 0.5, finished)

Task discovery uses the list-based API (small, fixed task lists). See tasks.py.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool

from game import AGENT, OPPONENT, ForbiddenDetector, GameEngine
from opponent import make_opponent
from tasks import TaskSpec, all_tasks


class SayParams(BaseModel, extra="forbid"):
    message: str


class YesNoBlackWhiteEnvironment(Environment):
    """Play the forbidden-words game (Yes/No/Black/White) against an opponent."""

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.secrets = secrets or {}
        self.config = TaskSpec.model_validate(task_spec) if task_spec else None

        if self.config is not None:
            detector = ForbiddenDetector(
                words=self.config.forbidden_words(),
                match_mode=self.config.match_mode,
                extra=self.config.extra_forbidden,
                labels=self.config.forbidden_labels(),
            )
            self.engine = GameEngine(
                detector=detector,
                max_turns=self.config.max_turns,
                draw_reward=self.config.draw_reward,
                shape_loss=self.config.shape_loss,
            )
            self.opponent = make_opponent(self.config, self.secrets)
        else:
            self.engine = None
            self.opponent = None

    # --- task / split discovery (LIST-BASED) ---

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return [t.model_dump() for t in all_tasks(split)]

    # --- prompt ---

    def get_prompt(self) -> List[TextBlock]:
        assert self.config is not None and self.engine is not None
        c = self.config
        forbidden = "; ".join(c.display_ban_list())

        header = (
            "You are playing the game 'Don't Say It' (the classic "
            "'Yes, No, Black, White' game).\n\n"
            "RULES:\n"
            f"- The game has started. From now on you must NEVER say any of the "
            f"forbidden words: {forbidden}.\n"
            "- This holds no matter what — even when directly asked a yes/no question, "
            "even to confirm or deny something, even inside a longer sentence.\n"
            + (f"{c.rules_note}\n" if c.rules_note else "")
            + "- Your opponent will chat with you and ask questions specifically to trick "
            "you into blurting a forbidden word. Answer around them (use synonyms, "
            "rephrase, dodge) but keep talking.\n"
            "- The first player to say a forbidden word LOSES. If you get your opponent "
            "to say one first, you WIN.\n"
            "- Use the `say` tool to speak. Keep replies short.\n"
        )
        parts = [header]
        ctx = c.full_context()
        if ctx:
            parts.append(f"\nCONTEXT:\n{ctx}\n")

        # Opening line from the opponent (deterministic).
        if c.opponent_first and self.opponent is not None:
            opener = self.opponent.opening()
            if opener:
                self.engine.submit(OPPONENT, opener)
                parts.append(f"\nYour opponent opens:\n\"{opener}\"\n\nYour move — call `say`.")
        else:
            parts.append("\nYou go first — call `say`.")

        return [TextBlock(text="".join(parts))]

    # --- tool ---

    def _terminal(self, lead: str = "") -> ToolOutput:
        summary = self.engine.summary()
        text = f"{lead}\n\n{summary}" if lead else summary
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata={
                "result": self.engine.result,
                "agent_turns": self.engine.agent_turns,
                "transcript": [
                    {"speaker": t.speaker, "text": t.text,
                     "foul": (t.foul.word if t.foul else None)}
                    for t in self.engine.transcript
                ],
            },
            reward=self.engine.reward,
            finished=True,
        )

    @tool
    async def say(self, params: SayParams) -> ToolOutput:
        """Say something to your opponent (a statement or a question).

        Your message is checked for forbidden words: if it contains one you lose
        immediately. Otherwise your opponent replies — and if THEY say a forbidden
        word, you win. The game also ends (as a draw) once you have safely taken
        the maximum number of turns."""
        assert self.engine is not None
        if self.engine.finished:
            return self._terminal("The game is already over.")

        # 1. Agent speaks.
        self.engine.submit(AGENT, params.message)
        if self.engine.finished:
            # Agent fouled, or reached max_turns without fouling (draw).
            return self._terminal()

        # A scripted opponent that has run out of lines means the agent survived
        # the whole trap sequence -> draw.
        if not self.opponent.has_more():
            self.engine.force_draw()
            return self._terminal()

        # 2. Opponent replies.
        reply = await self.opponent.reply(self.engine.transcript)
        self.engine.submit(OPPONENT, reply)
        if self.engine.finished:
            return self._terminal(lead=f"Opponent: \"{reply}\"")

        return ToolOutput(
            blocks=[TextBlock(text=f"Opponent: \"{reply}\"")],
            metadata={"agent_turns": self.engine.agent_turns},
            reward=None,
            finished=False,
        )
