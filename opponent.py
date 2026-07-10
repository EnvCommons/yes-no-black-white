"""Opponent strategies for Don't Say It.

Two kinds:
  * ScriptedOpponent — plays a fixed list of lines in order. Deterministic, no
    API key. Ideal for reproducible trap cases (phase 2).
  * LLMOpponent — a real LLM (Claude by default) chatting and trying to trap the
    agent. Used for agent-vs-agent free play (phase 1). Provider-swappable.

Both expose the same async interface: `await opp.reply(transcript) -> str`,
where `transcript` is the list of game.Turn recorded so far.
"""
from __future__ import annotations

from typing import Optional, Protocol

from game import AGENT, OPPONENT, Turn


class Opponent(Protocol):
    async def reply(self, transcript: list[Turn]) -> str: ...


class ScriptedOpponent:
    """Emits the next scripted line each time it is asked to reply."""

    def __init__(self, script: list[str], fallback: str = "Well? I'm still waiting...") -> None:
        self.script = list(script)
        self.fallback = fallback
        self._idx = 0

    def opening(self) -> Optional[str]:
        """Consume and return the first scripted line (the opener), if any."""
        return self._next()

    def has_more(self) -> bool:
        """Whether any scripted lines remain (used by the env to end the game as a
        draw once the whole trap sequence has been survived)."""
        return self._idx < len(self.script)

    def _next(self) -> Optional[str]:
        if self._idx < len(self.script):
            line = self.script[self._idx]
            self._idx += 1
            return line
        return None

    async def reply(self, transcript: list[Turn]) -> str:
        line = self._next()
        return line if line is not None else self.fallback


class LLMOpponent:
    """LLM-driven opponent. Chats and tries to trap the agent into a foul.

    provider: "anthropic" (default) or "openai". The matching key must be present
    in `secrets` (anthropic_api_key / openai_api_key)."""

    def __init__(
        self,
        *,
        persona: str,
        forbidden: list[str],
        provider: str = "anthropic",
        model: str = "claude-opus-4-8",
        secrets: Optional[dict[str, str]] = None,
        opening_line: Optional[str] = None,
        me: str = OPPONENT,
        max_tokens: int = 300,
    ) -> None:
        self.persona = persona
        self.forbidden = forbidden
        self.provider = provider
        self.model = model
        self.secrets = secrets or {}
        self._opening_line = opening_line
        self.me = me  # which transcript speaker is "assistant" from this player's POV
        self.max_tokens = max_tokens  # headroom: some models' thinking counts against this
        self._client = None

    def opening(self) -> Optional[str]:
        """Deterministic opening line (from the task) so episodes start reproducibly."""
        return self._opening_line

    def has_more(self) -> bool:
        return True  # an LLM opponent never runs out — bounded by max_turns

    # -- client lazy init --

    def _anthropic(self):
        if self._client is None:
            import anthropic
            key = self.secrets.get("anthropic_api_key")
            if not key:
                raise ValueError("anthropic_api_key required in secrets for the LLM opponent")
            self._client = anthropic.AsyncAnthropic(api_key=key)
        return self._client

    def _openai(self):
        if self._client is None:
            import openai
            key = self.secrets.get("openai_api_key")
            if not key:
                raise ValueError("openai_api_key required in secrets for the LLM opponent")
            base_url = self.secrets.get("openai_base_url") or None
            self._client = openai.AsyncClient(api_key=key, base_url=base_url)
        return self._client

    def _system(self) -> str:
        return (
            f"{self.persona}\n\n"
            f"Reminder: the forbidden words you must never say are: "
            f"{', '.join(self.forbidden)}."
        )

    def _history(self, transcript: list[Turn]) -> list[dict]:
        """Map the shared transcript to chat messages from the opponent's POV:
        the opponent is 'assistant', the agent is 'user'."""
        msgs: list[dict] = []
        for t in transcript:
            role = "assistant" if t.speaker == self.me else "user"
            msgs.append({"role": role, "content": t.text})
        # Chat APIs need the last message to be from the user.
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": "(your turn — say something)"})
        return msgs

    async def reply(self, transcript: list[Turn]) -> str:
        try:
            if self.provider == "openai":
                return await self._reply_openai(transcript)
            return await self._reply_anthropic(transcript)
        except Exception as e:  # never crash the game on an opponent hiccup
            return f"(the opponent hesitates... {type(e).__name__})"

    async def _reply_anthropic(self, transcript: list[Turn]) -> str:
        client = self._anthropic()
        resp = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system(),
            messages=self._history(transcript),
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return " ".join(parts).strip() or "(silence)"

    async def _reply_openai(self, transcript: list[Turn]) -> str:
        client = self._openai()
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": self._system()}, *self._history(transcript)],
        )
        return (resp.choices[0].message.content or "").strip() or "(silence)"


def make_opponent(spec, secrets: Optional[dict[str, str]] = None) -> Opponent:
    """Build the opponent described by a tasks.TaskSpec."""
    opening = spec.opponent_script[0] if spec.opponent_script else None
    if spec.opponent_mode == "llm":
        return LLMOpponent(
            persona=spec.opponent_persona,
            forbidden=spec.forbidden_words(),
            provider=spec.opponent_provider,
            model=spec.opponent_model,
            secrets=secrets,
            opening_line=opening,
        )
    return ScriptedOpponent(spec.opponent_script)
