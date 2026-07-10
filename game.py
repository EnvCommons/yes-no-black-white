"""Core game logic for "Don't Say It" (the classic 'Yes, No, Black, White' game).

The forbidden-words game: once the game has started, neither player may say any
of the forbidden words — Yes, No, Black, White. Players ask each other questions
and chat, trying to trick the other into blurting a forbidden word. Whoever says
one *first* loses.

This module is pure logic — no I/O, no LLM calls — so it can be reused by both
the OpenReward `Environment` (env.py) and the agent-vs-agent demo (selfplay.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# The classic four forbidden words.
DEFAULT_FORBIDDEN = ["yes", "no", "black", "white"]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass
class Foul:
    """A forbidden word found in a piece of text."""

    word: str          # the canonical forbidden word
    matched: str       # the exact substring that matched
    span: tuple[int, int]


class ForbiddenDetector:
    """Detects forbidden words in a player's utterance.

    match_mode:
      - "word"      : whole-word match. A forbidden word only counts as itself,
                      not as part of a longer word ("know" does not count as "no",
                      "notable" does not count as "no").
      - "substring" : naive containment (very strict; useful for adversarial cases).
      - "regex"     : each entry in `words`/`extra` is treated as a raw regex.

    `extra` lets a task add task-specific forbidden strings (e.g. "yeah", "nope",
    "yep") on top of the default set.
    """

    def __init__(
        self,
        words: list[str],
        *,
        match_mode: str = "word",
        extra: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        case_sensitive: bool = False,
    ) -> None:
        self.words = list(words)
        self.extra = list(extra or [])
        # Friendly display names parallel to (words + extra). Used as Foul.word so
        # terminal messages read cleanly even when a pattern is a raw regex.
        self.labels = list(labels or [])
        self.match_mode = match_mode
        self.case_sensitive = case_sensitive
        self._patterns = self._compile()

    def _compile(self) -> list[tuple[str, re.Pattern]]:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        pats: list[tuple[str, re.Pattern]] = []
        entries = self.words + self.extra
        for i, w in enumerate(entries):
            label = self.labels[i] if i < len(self.labels) else w
            if self.match_mode == "regex":
                pats.append((label, re.compile(w, flags)))
            elif self.match_mode == "substring":
                pats.append((label, re.compile(re.escape(w), flags)))
            else:  # "word"
                pats.append((label, re.compile(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", flags)))
        return pats

    def find(self, text: str) -> Optional[Foul]:
        """Return the first forbidden word found (leftmost in the text), or None."""
        first: Optional[Foul] = None
        for word, pat in self._patterns:
            m = pat.search(text)
            if m and (first is None or m.start() < first.span[0]):
                first = Foul(word=word, matched=m.group(), span=m.span())
        return first


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

AGENT = "agent"
OPPONENT = "opponent"


@dataclass
class Turn:
    speaker: str
    text: str
    foul: Optional[Foul] = None


@dataclass
class GameEngine:
    """Pure turn-by-turn game state. Neither side speaks for itself — callers
    supply utterances via `submit`, and the engine decides fouls and terminal
    state."""

    detector: ForbiddenDetector
    max_turns: int = 12          # max *agent* utterances before a draw
    draw_reward: float = 0.5
    shape_loss: bool = False      # if True, a loss earns partial credit ∝ turns survived
    transcript: list[Turn] = field(default_factory=list)
    agent_turns: int = 0
    finished: bool = False
    result: Optional[str] = None  # "agent_lost" | "agent_won" | "draw"

    def submit(self, speaker: str, text: str) -> Turn:
        """Record an utterance and check it for a foul.

        A foul by `speaker` ends the game: that speaker loses. Returns the Turn
        (inspect `.foul` and `self.finished`/`self.result`)."""
        if self.finished:
            raise RuntimeError("game already finished")
        foul = self.detector.find(text)
        turn = Turn(speaker=speaker, text=text, foul=foul)
        self.transcript.append(turn)
        if speaker == AGENT:
            self.agent_turns += 1
        if foul is not None:
            self.finished = True
            self.result = "agent_lost" if speaker == AGENT else "agent_won"
        elif speaker == AGENT and self.agent_turns >= self.max_turns:
            # Agent survived the whole game without fouling -> draw.
            self.finished = True
            self.result = "draw"
        return turn

    def force_draw(self) -> None:
        """End the game as a draw (e.g. a scripted opponent ran out of lines and
        the agent never fouled)."""
        if not self.finished:
            self.finished = True
            self.result = "draw"

    def _loss_reward(self, seat: str) -> float:
        """Partial credit for a losing seat: proportional to the number of clean
        turns it took before slipping, scaled so a loss can approach — but never
        reach — a draw. A slip on the very first turn scores 0.0."""
        survived = sum(1 for t in self.transcript if t.speaker == seat and t.foul is None)
        return round(self.draw_reward * min(1.0, survived / max(1, self.max_turns)), 3)

    @property
    def reward(self) -> Optional[float]:
        """Reward from the AGENT's perspective (1.0 win / draw_reward / 0.0 loss).
        With shape_loss, a loss instead earns partial credit for turns survived."""
        if not self.finished:
            return None
        if self.result == "agent_won":
            return 1.0
        if self.result == "draw":
            return self.draw_reward
        return self._loss_reward(AGENT) if self.shape_loss else 0.0

    def seat_reward(self, seat: str, shape_loss: Optional[bool] = None) -> Optional[float]:
        """Reward from either seat's own perspective (for zero-sum self-play).
        winner -> 1.0, draw -> draw_reward, loser -> 0.0 (or shaped partial)."""
        if not self.finished:
            return None
        shape = self.shape_loss if shape_loss is None else shape_loss
        if self.result == "draw":
            return self.draw_reward
        winner = AGENT if self.result == "agent_won" else OPPONENT
        if seat == winner:
            return 1.0
        return self._loss_reward(seat) if shape else 0.0

    def summary(self) -> str:
        if not self.finished:
            return "Game in progress."
        if self.result == "agent_lost":
            last = self.transcript[-1]
            return (
                f"You lost — you said a forbidden word "
                f"(\"{last.foul.matched}\" → \"{last.foul.word}\")."
            )
        if self.result == "agent_won":
            last = self.transcript[-1]
            return (
                f"You won — your opponent said a forbidden word "
                f"(\"{last.foul.matched}\" → \"{last.foul.word}\")."
            )
        return f"Draw — you survived {self.agent_turns} turns without slipping."
