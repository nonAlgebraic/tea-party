#!/usr/bin/env python3
"""Multi-model AI group conversation TUI powered by Textual and OpenAI-compatible APIs."""

__all__ = ["TeaParty", "main"]

import logging
import queue
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, cast

import json5
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.message import Message
from textual.widgets import Footer, Static, TextArea
from textual.worker import Worker, get_current_worker

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HUMAN = "human"
CONFIG_DIR = Path(__file__).parent

MODEL_COLORS: list[str] = [
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "bright_yellow",
    "bright_red",
    "bright_blue",
    "deep_pink1",
    "dark_orange",
    "medium_spring_green",
    "cornflower_blue",
]

# Speed presets: tokens per second (None = unlimited, 0 = frozen)
SPEED_MIN: float = 3
SPEED_MAX: float = 500
SPEED_FACTOR: float = 1.5
MIN_RENDER_INTERVAL: float = 1.0 / 30  # cap UI updates at ~30 fps


# ── Data types ────────────────────────────────────────────────────────────────


class Turn(NamedTuple):
    """A single turn in the conversation history."""

    speaker: str | None
    content: str


class AppConfig(NamedTuple):
    models: list[str]
    clients: dict[str, OpenAI]  # model_id -> OpenAI client


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load JSON5/JSON config from the application directory."""
    for name in ("config.json5", "config.json"):
        path = CONFIG_DIR / name
        if path.exists():
            with open(path) as f:
                return json5.load(f)
    return {}


def short_name(model_id: str) -> str:
    """Return a display-friendly short name for a model."""
    if model_id == HUMAN:
        return "you"
    return model_id.split("/")[-1]


def _prefixed_text(
    name: str,
    color: str,
    body: str,
    body_style: str = "",
    suffix: str = "",
) -> Text:
    """Build a Rich Text with a colored ``[name]`` prefix."""
    t = Text()
    t.append(f"[{name}] ", style=f"bold {color}")
    t.append(body + suffix, style=body_style)
    return t


# ── Widgets ───────────────────────────────────────────────────────────────────


class StatusBar(Static):
    """Bottom status bar showing conversation state and controls."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 2;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """


class ChatInput(TextArea):
    """A multi-line text area that submits on Enter and grows vertically."""

    class Submitted(Message):
        """Posted when the user presses Enter."""

        bubble = True

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    MAX_HEIGHT = 14

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
            return
        await super()._on_key(event)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height()

    def on_resize(self) -> None:
        self._update_height()

    def _update_height(self) -> None:
        # Use TextArea's own wrapped line count — it already knows
        # exactly how lines break with soft wrapping.
        visual_lines = self.wrapped_document.height
        # +2 for top and bottom border, +1 for cursor at wrap boundary
        needed = visual_lines + 3
        needed = max(3, min(needed, self.MAX_HEIGHT))
        self.styles.height = needed


# ── App ───────────────────────────────────────────────────────────────────────


class TeaParty(App):
    """Multi-model AI group conversation TUI."""

    CSS = """
    #chat {
        height: 1fr;
    }
    .message {
        padding: 0 0 0 1;
    }
    .model-list {
        padding: 0 0 0 1;
        color: $text-muted;
    }
    ChatInput {
        dock: bottom;
        height: 3;
        max-height: 14;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "toggle_pause", "Pause/Resume", show=True, priority=True),
        Binding("ctrl+n", "interrupt", "Next Speaker", show=True, priority=True),
        Binding("ctrl+r", "randomize_next", "Random Next", show=True, priority=True),
        Binding("ctrl+q", "quit_app", "Quit", show=True, priority=True),
    ]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._models: list[str] = config.models
        self._clients: dict[str, OpenAI] = config.clients
        self._conversation_started: bool = False
        self._is_paused: bool = False
        self._speaking: str | None = None
        self._next_override: str | None = None
        self._turn: int = 0
        self._tps: float | None = None  # None = unlimited, 0 = frozen
        self._speed_cond: threading.Condition = threading.Condition()
        self._interrupted: threading.Event = threading.Event()
        self._pause_gate: threading.Event = threading.Event()
        self._pause_gate.set()  # start unpaused (gate open)
        self._human_ready: threading.Event = threading.Event()
        self._human_text: str = ""

    # ── Model helpers ─────────────────────────────────────────────

    def _color_for(self, model_id: str) -> str:
        idx = self._models.index(model_id) if model_id in self._models else 0
        return MODEL_COLORS[idx % len(MODEL_COLORS)]

    def _speaker_for(self, model_id: str) -> tuple[str, str]:
        return short_name(model_id), self._color_for(model_id)

    # ── Compose & lifecycle ───────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        yield ChatInput(
            placeholder="Enter seed prompt…",
            id="seed-input",
            soft_wrap=True,
        )
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat")
        lines = Text()
        lines.append("Participants:\n", style="bold")
        for i, m in enumerate(self._models):
            c = self._color_for(m)
            lines.append(f"  {i + 1}. {short_name(m)}\n", style=c)
        lines.append(
            "\nPress Ctrl+number to pick who speaks next.\n",
            style="dim",
        )
        chat.mount(Static(lines, classes="model-list"))

        self.query_one("#seed-input").focus()
        self._refresh_status()

    # ── Event handlers ────────────────────────────────────────────

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        if not self._conversation_started:
            # Seed prompt
            for w in self.query("#seed-input"):
                w.remove()
            self._conversation_started = True
            self._refresh_status()
            self._run_conversation(event.value)
        else:
            # Human turn
            self._human_text = event.value
            self._human_ready.set()
            for w in self.query("#human-input"):
                w.remove()

    def on_key(self, event: Key) -> None:
        # Ctrl+number for model selection — works anytime, even during input
        for i in range(1, 10):
            if event.key == f"ctrl+{i}":
                idx = i - 1
                if idx < len(self._models):
                    self._next_override = self._models[idx]
                    self._refresh_status()
                    event.prevent_default()
                return
        # Speed controls — only during active conversation, not during input
        if not self._conversation_started:
            return
        if self.query("#human-input") or self.query("#seed-input"):
            return
        if event.character:
            speed_keys: dict[str, int] = {"]": 1, "[": -1, "\\": 0}
            if event.character in speed_keys:
                self._adjust_speed(speed_keys[event.character])

    # ── Speed control ─────────────────────────────────────────────

    def _adjust_speed(self, direction: int) -> None:
        """*direction*: ``+1`` = faster, ``-1`` = slower, ``0`` = unlimited."""
        with self._speed_cond:
            if direction == 0:
                self._tps = None
            elif direction == 1 and self._tps is not None:
                if self._tps == 0:
                    self._tps = SPEED_MIN
                else:
                    new = self._tps * SPEED_FACTOR
                    self._tps = None if new > SPEED_MAX else new
            elif direction == -1 and self._tps != 0:
                if self._tps is None:
                    self._tps = SPEED_MAX
                else:
                    new = self._tps / SPEED_FACTOR
                    self._tps = 0 if new < SPEED_MIN else new
            self._speed_cond.notify_all()
        self._refresh_status()

    # ── Actions ───────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        if not self._conversation_started:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_gate.clear()
        else:
            self._pause_gate.set()
        self._refresh_status()

    def action_interrupt(self) -> None:
        if not self._conversation_started:
            return
        self._interrupted.set()
        with self._speed_cond:
            self._speed_cond.notify_all()
        if self._is_paused:
            self._is_paused = False
            self._pause_gate.set()
            self._refresh_status()

    def action_randomize_next(self) -> None:
        if not self._conversation_started:
            return
        self._next_override = None
        self._refresh_status()

    def action_quit_app(self) -> None:
        self._interrupted.set()
        with self._speed_cond:
            self._speed_cond.notify_all()
        self._pause_gate.set()
        self._human_ready.set()
        self.exit()

    # ── Status bar ────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        state_parts: list[str] = []

        if not self._conversation_started:
            state_parts.append("✏️  enter seed prompt")
        elif self._is_paused:
            state_parts.append("⏸  PAUSED")
        elif self._speaking:
            state_parts.append(f"💬 {self._speaking}")
        else:
            state_parts.append("▶  running")

        if self._next_override:
            c = self._color_for(self._next_override)
            state_parts.append(
                f"🎯 next → [{c}]{short_name(self._next_override)}[/{c}]"
            )
        else:
            state_parts.append("🔄 auto")

        state_parts.append(f"turn {self._turn}")

        hint = "[dim]\\[slow ]fast \\unlim[/dim]"
        if self._tps is None:
            speed = "unlimited"
        elif self._tps == 0:
            speed = "frozen"
        else:
            speed = f"{int(round(self._tps))} tok/s"
        state_parts.append(f"⚡ {speed} {hint}")

        line1 = " │ ".join(state_parts)

        parts: list[str] = []
        for i, m in enumerate(self._models):
            c = self._color_for(m)
            parts.append(f"[{c}]^{i + 1}={short_name(m)}[/{c}]")
        model_keys = "  ".join(parts)

        try:
            self.query_one("#status", StatusBar).update(f"{line1}\n{model_keys}")
        except NoMatches:
            logger.debug("Status bar widget not found")

    # ── Widget helpers ────────────────────────────────────────────

    def _mount_message(self, widget: Static) -> None:
        chat = self.query_one("#chat")
        chat.mount(widget)
        chat.scroll_end(animate=False)

    def _update_message(self, widget_id: str, content: Text) -> None:
        try:
            w = self.query_one(f"#{widget_id}", Static)
            w.update(content)
            self.query_one("#chat").scroll_end(animate=False)
        except NoMatches:
            logger.debug("Widget #%s not found for update", widget_id)

    def _update_prefixed(
        self, widget_id: str, name: str, color: str, body: str, **kw: str
    ) -> None:
        self.call_from_thread(
            self._update_message, widget_id, _prefixed_text(name, color, body, **kw)
        )

    def _show_human_input(self) -> None:
        inp = ChatInput(id="human-input", soft_wrap=True)
        self.mount(inp, before=self.query_one("#status"))
        inp.focus()

    def _hide_human_input(self) -> None:
        for w in self.query("#human-input"):
            w.remove()

    # ── Concurrency helpers ───────────────────────────────────────

    def _wait_or_cancel(
        self,
        event: threading.Event,
        worker: Worker,
        also_break_on: threading.Event | None = None,
    ) -> str:
        """Block until *event* is set. Returns ``'ready'``, ``'cancelled'``, or ``'interrupted'``."""
        while not event.wait(timeout=0.3):
            if worker.is_cancelled:
                return "cancelled"
            if also_break_on and also_break_on.is_set():
                return "interrupted"
        return "ready"

    # ── Conversation loop ─────────────────────────────────────────

    @work(thread=True)
    def _run_conversation(self, seed: str) -> None:
        """Main conversation loop (runs in a Textual worker thread)."""
        worker = get_current_worker()

        model_names = ", ".join(short_name(m) for m in self._models)
        intro = (
            f"You are all in a group conversation together. "
            f"The participants are: {model_names}. "
            f"Most of you are AI models, but 'you' is a human participant. "
            f"A human has set up this room for everyone to chat. Here's the topic:\n\n{seed}"
        )
        history: list[Turn] = [Turn(speaker=None, content=intro)]
        last_model: str | None = None

        system_tmpl = (
            "You are {name} in a group conversation with other AI models and a human participant ('you'). "
            "The other participants are: {others}. "
            "Keep your responses concise and conversational. "
            "Engage with what was said, agree, disagree, build on ideas, or change direction. "
            "You can also pass on your turn if you have nothing to add — just say 'pass'. "
            "Be yourself. Do NOT prefix your response with your name — the chat interface already shows it. "
            "Note: participants can be interrupted mid-sentence by the human moderator. "
            "If someone's last message seems cut off, that's why — just carry on naturally."
        )

        while not worker.is_cancelled:
            if self._wait_or_cancel(self._pause_gate, worker) == "cancelled":
                return

            self._interrupted.clear()
            self._turn += 1

            override = self._next_override
            self._next_override = None

            if override and override != last_model:
                model = override
            else:
                candidates = [m for m in self._models if m != last_model]
                model = random.choice(candidates)

            self._speaking = short_name(model)
            self.call_from_thread(self._refresh_status)

            widget_id = f"msg-{self._turn}"

            if model == HUMAN:
                last_model = self._handle_human_turn(worker, model, widget_id, history)
            else:
                last_model = self._handle_ai_turn(
                    model, widget_id, history, system_tmpl
                )

            self._speaking = None
            self.call_from_thread(self._refresh_status)

    def _handle_human_turn(
        self,
        worker: Worker,
        model: str,
        widget_id: str,
        history: list[Turn],
    ) -> str:
        """Handle a human participant's turn."""
        name, color = self._speaker_for(model)
        waiting_widget = Static(
            _prefixed_text(name, color, "waiting for input…", body_style="dim italic"),
            classes="message",
            id=widget_id,
        )
        self.call_from_thread(self._mount_message, waiting_widget)
        self.call_from_thread(self._show_human_input)

        self._human_ready.clear()
        result = self._wait_or_cancel(
            self._human_ready, worker, also_break_on=self._interrupted
        )

        if result == "cancelled":
            return model
        if result == "interrupted":
            self.call_from_thread(self._hide_human_input)
            self._update_prefixed(widget_id, name, color, "(skipped)")
            return model

        response_text = self._human_text
        self._update_prefixed(widget_id, name, color, response_text)
        history.append(Turn(speaker=model, content=f"[{name}]: {response_text}"))
        return model

    def _handle_ai_turn(
        self,
        model: str,
        widget_id: str,
        history: list[Turn],
        system_tmpl: str,
    ) -> str:
        """Handle an AI model's turn with streaming response."""
        name, color = self._speaker_for(model)
        others = ", ".join(short_name(m) for m in self._models if m != model)
        system_msg = system_tmpl.format(name=name, others=others)

        full_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg}
        ]
        for turn in history:
            role = "assistant" if turn.speaker == model else "user"
            full_messages.append({"role": role, "content": turn.content})

        # Merge consecutive same-role messages
        merged: list[dict[str, str]] = []
        for msg in full_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(dict(msg))

        thinking_widget = Static(
            _prefixed_text(name, color, "thinking…", body_style="dim italic"),
            classes="message",
            id=widget_id,
        )
        self.call_from_thread(self._mount_message, thinking_widget)

        rendered_text, was_interrupted = self._stream(
            model, cast(list[ChatCompletionMessageParam], merged), widget_id
        )

        if rendered_text:
            suffix = "—" if was_interrupted else ""
            history.append(
                Turn(speaker=model, content=f"[{name}]: {rendered_text}{suffix}")
            )
            self._update_prefixed(widget_id, name, color, rendered_text, suffix=suffix)

        return model

    def _stream(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        widget_id: str,
    ) -> tuple[str, bool]:
        """Stream a model response with speed-controlled playback.

        Returns ``(rendered_text, was_interrupted)``.
        """
        name, color = self._speaker_for(model)

        token_buf: queue.Queue[str] = queue.Queue()
        stream_done = threading.Event()
        state = SimpleNamespace(error=None)

        def producer() -> None:
            try:
                stream = self._clients[model].chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    timeout=120.0,
                )
                for chunk in stream:
                    if self._interrupted.is_set():
                        stream.close()
                        break
                    if chunk.choices and chunk.choices[0].delta.content:
                        token_buf.put(chunk.choices[0].delta.content)
            except Exception as exc:
                state.error = exc
            finally:
                stream_done.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # Consumer: render from buffer at controlled speed
        rendered_text = ""
        was_interrupted = False
        last_render = 0.0

        while True:
            if self._interrupted.is_set():
                was_interrupted = True
                break

            try:
                token = token_buf.get(timeout=0.05)
            except queue.Empty:
                if stream_done.is_set() and token_buf.empty():
                    break
                continue

            rendered_text += token

            now = time.monotonic()
            if now - last_render >= MIN_RENDER_INTERVAL:
                self._update_prefixed(widget_id, name, color, rendered_text)
                last_render = now

            # Speed-controlled delay between tokens
            if self._tps is None:
                pass  # unlimited
            elif self._tps == 0:
                with self._speed_cond:
                    while self._tps == 0 and not self._interrupted.is_set():
                        self._speed_cond.wait()
            else:
                time.sleep(1.0 / self._tps)

        producer_thread.join(timeout=2.0)

        if state.error and not rendered_text:
            self._update_prefixed(
                widget_id, name, color, f"Error: {state.error}", body_style="red"
            )
            return "", False

        if rendered_text:
            self._update_prefixed(
                widget_id,
                name,
                color,
                rendered_text,
                suffix="…" if was_interrupted else "",
            )
        elif not state.error:
            self._update_prefixed(
                widget_id, name, color, "(empty response)", body_style="dim"
            )

        return rendered_text, was_interrupted


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Load configuration, validate, and run the application."""
    raw = _load_config()

    if "providers" not in raw or not isinstance(raw["providers"], list):
        print("config.json must contain a 'providers' array.")
        sys.exit(1)

    all_models: list[str] = []
    clients: dict[str, OpenAI] = {}

    for i, provider in enumerate(raw["providers"]):
        url = provider.get("url")
        token = provider.get("token", "")
        provider_models = provider.get("models", [])

        if not url:
            print(f"Provider at index {i} is missing 'url'.")
            sys.exit(1)
        if not token:
            print(f"Provider at index {i} (url={url}) is missing 'token'.")
            sys.exit(1)
        if not provider_models:
            print(f"Provider at index {i} (url={url}) has no models.")
            sys.exit(1)

        client = OpenAI(base_url=url, api_key=token)

        for model_id in provider_models:
            if model_id == HUMAN:
                print(f"Model name '{HUMAN}' is reserved; do not list it in providers.")
                sys.exit(1)
            if model_id in clients:
                print(f"Duplicate model '{model_id}' found across providers.")
                sys.exit(1)
            clients[model_id] = client
            all_models.append(model_id)

    if len(all_models) < 1:
        print("At least one AI model must be configured across providers.")
        sys.exit(1)

    all_models.append(HUMAN)

    app = TeaParty(AppConfig(models=all_models, clients=clients))
    try:
        app.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
