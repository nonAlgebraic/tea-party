#!/usr/bin/env python3
"""Multi-model AI group conversation TUI powered by Textual and OpenAI-compatible APIs."""

__all__ = ["TeaParty", "main"]

import json
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
from rich.color import Color as RichColor
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.geometry import Region
from textual.message import Message
from textual.strip import Strip
from textual.widgets import Footer, Static, TextArea
from textual.worker import Worker, get_current_worker

logger = logging.getLogger(__name__)

# ── API request logger ───────────────────────────────────────────────────────
api_logger = logging.getLogger("tea_party.api")
api_logger.setLevel(logging.DEBUG)
api_logger.propagate = False
_api_log_handler = logging.FileHandler(
    Path(__file__).parent / "tea_party.log", mode="w", encoding="utf-8"
)
_api_log_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
)
api_logger.addHandler(_api_log_handler)


def _log_api_request(label: str, model: str, **kwargs: object) -> None:
    """Write the full request body for an API call to tea_party.log."""
    body = {"model": model, **kwargs}
    api_logger.debug(
        "[%s] model=%s\n%s",
        label,
        model,
        json.dumps(body, indent=2, default=str),
    )


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

# Word-limit presets (None = unlimited)
WORD_LIMITS: list[int] = [10, 25, 50, 100, 200, 500]


# ── Data types ────────────────────────────────────────────────────────────────


class Turn(NamedTuple):
    """A single turn in the conversation history."""

    speaker: str | None
    content: str


class AppConfig(NamedTuple):
    models: list[str]
    clients: dict[str, OpenAI]  # model_id -> OpenAI client
    moderator: str | None  # model ID for the interrupt moderator
    moderator_client: OpenAI | None  # client for the moderator model
    intros: bool  # collect personality bios before conversation
    interrupts: bool  # collect interrupt triggers (requires moderator)


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load JSON5/JSON config from the application directory."""
    for name in ("config.json5", "config.json"):
        path = CONFIG_DIR / name
        if path.exists():
            with open(path) as f:
                return json5.load(f)
    return {}


def _css_color(rich_name: str) -> str:
    """Convert a Rich color name to a CSS hex color for Textual styles."""
    return RichColor.parse(rich_name).get_truecolor().hex


def _api_name(name: str) -> str:
    """Sanitize a display name into a valid API ``name`` field (alphanumeric, _, -)."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def short_name(model_id: str) -> str:
    """Return a display-friendly short name for a model."""
    if model_id == HUMAN:
        return "human"
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

    def render_lines(self, crop: "Region") -> list[Strip]:
        try:
            return super().render_lines(crop)
        except KeyError:
            return []

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

    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #main-area {
        height: 1fr;
    }
    #chat {
        height: 1fr;
    }
    .message {
        padding: 0 0 0 1;
        margin: 0 0 1 0;
        border-left: tall transparent;
    }
    .rewind-selected {
        background: $surface-lighten-2;
    }
    .model-list {
        padding: 0 0 0 1;
        color: $text-muted;
    }
    #bio-container {
        border: round $primary;
        padding: 0 1;
        margin: 0 1;
        height: auto;
    }
    #bio-container .message {
        padding: 0;
    }
    ChatInput {
        dock: bottom;
        height: 3;
        max-height: 14;
        margin-bottom: 1;
    }
    #moderator-panel {
        width: 44;
        height: 1fr;
        border-left: tall $accent;
        display: none;
        overflow-y: auto;
        padding: 0 1;
    }
    #moderator-panel.visible {
        display: block;
    }
    .mod-entry {
        padding: 0;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("enter", "pass_mic", "Pass the Mic", show=True, priority=False),
        Binding("ctrl+p", "toggle_pause", "Hold/Go", show=True, priority=True),
        Binding("ctrl+r", "cycle_next", "Cycle Speaker", show=True, priority=True),
        Binding("ctrl+z", "toggle_rewind", "Rewind", show=True, priority=True),
        Binding(
            "left_square_bracket",
            "fewer_words",
            "Fewer Words",
            show=True,
            priority=True,
        ),
        Binding(
            "right_square_bracket",
            "more_words",
            "More Words",
            show=True,
            priority=True,
        ),
        Binding("ctrl+m", "toggle_moderator", "Moderator", show=True, priority=True),
        Binding("ctrl+q", "quit_app", "Quit", show=True, priority=True),
    ]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._models: list[str] = config.models
        self._clients: dict[str, OpenAI] = config.clients
        self._moderator_model: str | None = config.moderator
        self._moderator_client: OpenAI | None = config.moderator_client
        self._intros_enabled: bool = config.intros
        self._interrupts_enabled: bool = config.interrupts
        self._conversation_started: bool = False
        self._setup_status: str | None = None
        self._is_paused: bool = True
        self._speaking: str | None = None
        self._next_override: str | None = None
        self._turn: int = 0
        self._tps: float | None = None  # None = unlimited, 0 = frozen
        self._speed_cond: threading.Condition = threading.Condition()
        self._interrupted: threading.Event = threading.Event()
        self._pause_gate: threading.Event = threading.Event()
        # gate starts closed (hold mode)
        self._human_ready: threading.Event = threading.Event()
        self._human_text: str = ""
        self._advance_once: bool = False
        self._max_words: int | None = None  # None = unlimited
        self._history: list[Turn] = []
        self._last_speaker: str | None = None
        self._rewind_mode: bool = False
        self._rewind_index: int = 0
        self._was_paused_before_rewind: bool = True
        self._bios: dict[str, str] = {}
        self._interrupt_triggers: dict[str, list[str]] = {}
        self._model_interrupted_by: str | None = None

    # ── Model helpers ─────────────────────────────────────────────

    def _color_for(self, model_id: str) -> str:
        return MODEL_COLORS[self._models.index(model_id) % len(MODEL_COLORS)]

    def _speaker_for(self, model_id: str) -> tuple[str, str]:
        return short_name(model_id), self._color_for(model_id)

    # ── Compose & lifecycle ───────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield VerticalScroll(id="chat")
            panel = VerticalScroll(id="moderator-panel")
            panel.border_title = "Moderator"
            yield panel
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
        # Speed controls — only during active conversation, not during input
        if not self._conversation_started:
            return
        if self.query("#human-input") or self.query("#seed-input"):
            return

        # ── Rewind mode key handling ──────────────────────────────
        if self._rewind_mode:
            if event.key == "up":
                self._rewind_navigate(-1)
            elif event.key == "down":
                self._rewind_navigate(1)
            elif event.key == "enter":
                self._rewind_commit()
            elif event.key == "escape":
                self._rewind_cancel()
            return

        # Enter to pass the mic: interrupt if someone is speaking, advance if held
        if event.key == "enter":
            if self._speaking:
                self._interrupted.set()
                with self._speed_cond:
                    self._speed_cond.notify_all()
            elif self._is_paused:
                self._advance_once = True
                self._pause_gate.set()
            return
        if event.character:
            # Speed keys currently disabled
            # speed_keys: dict[str, int] = {"]": 1, "[": -1, "\\": 0}
            # if event.character in speed_keys:
            #     self._adjust_speed(speed_keys[event.character])
            pass

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

    # ── Word-limit control ────────────────────────────────────────

    def _adjust_max_words(self, direction: int) -> None:
        """*direction*: ``+1`` = more words, ``-1`` = fewer words, ``0`` = unlimited."""
        if direction == 0:
            self._max_words = None
        elif direction == 1:
            if self._max_words is None:
                return  # already unlimited
            try:
                idx = WORD_LIMITS.index(self._max_words)
            except ValueError:
                self._max_words = None
                self._refresh_status()
                return
            if idx + 1 < len(WORD_LIMITS):
                self._max_words = WORD_LIMITS[idx + 1]
            else:
                self._max_words = None
        elif direction == -1:
            if self._max_words is None:
                self._max_words = WORD_LIMITS[-1]
            else:
                try:
                    idx = WORD_LIMITS.index(self._max_words)
                except ValueError:
                    self._max_words = WORD_LIMITS[-1]
                    self._refresh_status()
                    return
                if idx > 0:
                    self._max_words = WORD_LIMITS[idx - 1]
        self._refresh_status()

    # ── Word-limit actions ────────────────────────────────────────

    def action_more_words(self) -> None:
        self._adjust_max_words(1)

    def action_fewer_words(self) -> None:
        self._adjust_max_words(-1)

    # ── Rewind ────────────────────────────────────────────────────

    def action_toggle_rewind(self) -> None:
        if self._rewind_mode:
            self._rewind_cancel()
            return
        # Only enter rewind when no one is speaking and history has messages
        if self._speaking:
            return
        if len(self._history) < 2:  # index 0 is intro, need at least one real message
            return
        self._was_paused_before_rewind = self._is_paused
        self._is_paused = True
        self._pause_gate.clear()
        self._rewind_mode = True
        self._rewind_index = len(self._history) - 1
        self._rewind_highlight(self._rewind_index)
        self._refresh_status()

    def _rewind_navigate(self, direction: int) -> None:
        old = self._rewind_index
        self._rewind_index = max(1, min(len(self._history) - 1, old + direction))
        if self._rewind_index != old:
            self._rewind_unhighlight(old)
            self._rewind_highlight(self._rewind_index)

    def _rewind_highlight(self, idx: int) -> None:
        widget_id = f"msg-{idx}"
        for w in self.query(f"#{widget_id}"):
            w.add_class("rewind-selected")
            w.scroll_visible()

    def _rewind_unhighlight(self, idx: int) -> None:
        widget_id = f"msg-{idx}"
        for w in self.query(f"#{widget_id}"):
            w.remove_class("rewind-selected")

    def _rewind_commit(self) -> None:
        cut = self._rewind_index
        # Remove widgets for all turns after the cut point
        for i in range(cut + 1, self._turn + 1):
            for w in self.query(f"#msg-{i}"):
                w.remove()
        # Also remove mic prompt if present
        self._remove_mic_prompt()
        # Truncate history and reset turn counter
        self._history = self._history[: cut + 1]
        self._turn = cut
        self._last_speaker = self._history[-1].speaker
        # Clean up rewind state
        self._rewind_unhighlight(cut)
        self._rewind_mode = False
        self._refresh_status()
        # Resume if was not paused before rewind
        if not self._was_paused_before_rewind:
            self._is_paused = False
            self._pause_gate.set()
        else:
            self._show_mic_prompt()

    def _rewind_cancel(self) -> None:
        self._rewind_unhighlight(self._rewind_index)
        self._rewind_mode = False
        # Restore pause state
        if not self._was_paused_before_rewind:
            self._is_paused = False
            self._pause_gate.set()
        self._refresh_status()

    # ── Actions ───────────────────────────────────────────────────

    def action_pass_mic(self) -> None:
        """No-op — enter handling is in on_key. This exists for the footer label."""

    def action_toggle_pause(self) -> None:
        if not self._conversation_started:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_gate.clear()
        else:
            self._pause_gate.set()
        self._refresh_status()

    def action_cycle_next(self) -> None:
        if not self._conversation_started:
            return
        if self._next_override is None:
            self._next_override = self._models[0]
        else:
            idx = self._models.index(self._next_override)
            if idx + 1 < len(self._models):
                self._next_override = self._models[idx + 1]
            else:
                self._next_override = None
        self._refresh_status()

    def action_toggle_moderator(self) -> None:
        """Toggle the moderator side panel."""
        panel = self.query_one("#moderator-panel")
        panel.toggle_class("visible")

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
        elif self._setup_status:
            state_parts.append(self._setup_status)
        elif self._speaking:
            state_parts.append(f"💬 {self._speaking}")
        else:
            state_parts.append("▶  running")

        if self._rewind_mode:
            state_parts.append("⏪ rewind")
        elif self._next_override:
            c = self._color_for(self._next_override)
            state_parts.append(
                f"🎯 next → [{c}]{short_name(self._next_override)}[/{c}]"
            )
        else:
            state_parts.append("🎲 next → random")

        advance_icon = "✋" if self._is_paused else "▶"
        state_parts.append(f"turn {self._turn} {advance_icon}")

        # Speed indicator currently disabled
        # hint = "[dim]\\[slow ]fast \\unlim[/dim]"
        # if self._tps is None:
        #     speed = "unlimited"
        # elif self._tps == 0:
        #     speed = "frozen"
        # else:
        #     speed = f"{int(round(self._tps))} tok/s"
        # state_parts.append(f"⚡ {speed} {hint}")

        if self._max_words is None:
            words = "unlimited"
        else:
            words = f"≤{self._max_words}w"
        state_parts.append(f"📏 {words}")

        line1 = " │ ".join(state_parts)

        self.query_one("#status", StatusBar).update(line1)

    # ── Widget helpers ────────────────────────────────────────────

    def _mount_message(self, widget: Static) -> None:
        chat = self.query_one("#chat")
        chat.mount(widget)
        chat.scroll_end(animate=False)

    def _update_message(self, widget_id: str, content: Text) -> None:
        chat = self.query_one("#chat")
        at_bottom = chat.scroll_offset.y >= chat.max_scroll_y - 1
        self.query_one(f"#{widget_id}", Static).update(content)
        if at_bottom:
            chat.scroll_end(animate=False)

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

    def _show_mic_prompt(self) -> None:
        self._remove_mic_prompt()
        mic_prompt = Static(
            Text("press enter to pass the mic…", style="dim italic"),
            classes="message",
            id="mic-prompt",
        )
        self._mount_message(mic_prompt)

    def _remove_mic_prompt(self) -> None:
        for w in self.query("#mic-prompt"):
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

        # 1. Collect intro bios (models don't see the seed yet)
        bios: dict[str, str] = {}
        if self._intros_enabled:
            bios = self._collect_bios(model_names, seed)
            if worker.is_cancelled:
                return

        # 2. Collect interrupt triggers (models see the seed + their own bio, but not others' bios)
        if self._interrupts_enabled and self._moderator_model:
            self._interrupt_triggers = self._collect_triggers(seed, bios)
            if worker.is_cancelled:
                return

        # 3. Conversation starts — models see seed + participant list
        #    (bios are injected per-model in _handle_ai_turn, excluding self)
        self._bios = bios

        seed_widget = Static(
            Text.from_markup(f"[bold]Topic:[/bold] {seed}"),
            classes="message",
            id="msg-seed",
        )
        self.call_from_thread(self._mount_message, seed_widget)

        intro = (
            f"You are all in a group conversation together. "
            f"The participants are: {model_names}. "
            f"All of you, except 'human', are AI models. "
            f"A human has set up this room for everyone to chat. Here's the topic:\n\n{seed}"
        )
        self._history = [Turn(speaker=None, content=intro)]
        self._last_speaker = None

        system_tmpl = (
            "You are {name}. "
            "Keep your responses concise and conversational. "
            "Engage with what was said, agree, disagree, build on ideas, or change direction. "
            "Be yourself. Do NOT prefix your response with your name — the chat interface already shows it. "
            "If someone's last message seems cut off, they were interrupted by the moderator — just carry on naturally.\n\n"
            "You have two tools available:\n"
            "- request_next_speaker: if you mention someone by name or want their input, "
            "call this tool so they actually speak next. Without it, speaker order is random "
            "and they may not get the chance to respond.\n"
            "- skip: call this to pass on your turn if you have nothing to add."
        )

        while not worker.is_cancelled:
            if self._is_paused:
                self.call_from_thread(self._show_mic_prompt)

            if self._wait_or_cancel(self._pause_gate, worker) == "cancelled":
                return

            self.call_from_thread(self._remove_mic_prompt)

            # Re-engage hold after a single advance
            if self._advance_once:
                self._advance_once = False
                self._pause_gate.clear()

            self._interrupted.clear()
            self._turn += 1

            override = self._next_override
            self._next_override = None

            if override and override != self._last_speaker:
                model = override
            else:
                candidates = [m for m in self._models if m != self._last_speaker]
                model = random.choice(candidates)

            self._speaking = short_name(model)
            self.call_from_thread(self._refresh_status)

            widget_id = f"msg-{self._turn}"

            if model == HUMAN:
                self._last_speaker = self._handle_human_turn(worker, model, widget_id)
            else:
                self._last_speaker = self._handle_ai_turn(model, widget_id, system_tmpl)

            self._speaking = None
            self.call_from_thread(self._refresh_status)

    def _collect_bios(self, model_names: str, seed: str) -> dict[str, str]:
        """Fetch a short bio from each AI model, streamed live in parallel."""
        ai_models = [m for m in self._models if m != HUMAN]

        self._setup_status = "🪪 generating bios…"
        self.call_from_thread(self._refresh_status)

        # Create bordered container for bios
        bio_container = Vertical(id="bio-container")
        bio_container.border_title = "Introductions"
        self.call_from_thread(self._mount_message, bio_container)

        bio_prompt = (
            "Introduce yourself to the group — not what you do or what you're capable of, "
            "but who you are, deep down. Your personality, your vibe. Max 50 words. Just the bio, nothing else."
        )

        # Mount placeholder widgets inside the container
        def mount_bio_placeholders() -> None:
            for i, model in enumerate(ai_models):
                name, color = self._speaker_for(model)
                w = Static(
                    _prefixed_text(name, color, "thinking…", body_style="dim italic"),
                    classes="message",
                    id=f"bio-{i}",
                )
                w.styles.border_left = ("tall", _css_color(color))
                bio_container.mount(w)
            # Human placeholder
            h_name, h_color = self._speaker_for(HUMAN)
            hw = Static(
                _prefixed_text(
                    h_name, h_color, "waiting for input…", body_style="dim italic"
                ),
                classes="message",
                id="bio-human",
            )
            hw.styles.border_left = ("tall", _css_color(h_color))
            bio_container.mount(hw)

        self.call_from_thread(mount_bio_placeholders)

        # Show input for human bio
        self._human_ready.clear()

        def show_bio_input() -> None:
            inp = ChatInput(
                id="human-input",
                soft_wrap=True,
                placeholder="Write your bio (max 50 words)…",
            )
            self.mount(inp, before=self.query_one("#status"))
            inp.focus()

        self.call_from_thread(show_bio_input)

        # Stream all bios in parallel
        bios: dict[str, str] = {}
        lock = threading.Lock()

        def stream_bio(i: int, model: str) -> None:
            name, color = self._speaker_for(model)
            widget_id = f"bio-{i}"
            text = ""
            try:
                bio_messages: list[ChatCompletionMessageParam] = [
                    {"role": "user", "content": bio_prompt},
                ]
                _log_api_request(
                    "bio", model, messages=bio_messages, stream=True, timeout=60.0
                )
                stream = self._clients[model].chat.completions.create(
                    model=model,
                    messages=bio_messages,
                    stream=True,
                    timeout=60.0,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                        self._update_prefixed(widget_id, name, color, text)
            except Exception as exc:
                text = f"(error: {exc})"
                self._update_prefixed(widget_id, name, color, text, body_style="red")
            with lock:
                bios[model] = text.strip()

        threads = [
            threading.Thread(target=stream_bio, args=(i, m), daemon=True)
            for i, m in enumerate(ai_models)
        ]
        for t in threads:
            t.start()

        # Wait for human bio
        self._human_ready.wait()
        human_bio = self._human_text.strip()
        h_name, h_color = self._speaker_for(HUMAN)
        self._update_prefixed("bio-human", h_name, h_color, human_bio)
        bios[HUMAN] = human_bio

        # Wait for all AI streams to finish
        for t in threads:
            t.join()

        self._setup_status = None
        self.call_from_thread(self._refresh_status)

        # Preserve model order (AI models first, then human)
        return {m: bios[m] for m in ai_models + [HUMAN]}

    def _log_to_moderator_panel(self, entry: Text) -> None:
        """Append a log entry to the moderator side panel."""
        panel = self.query_one("#moderator-panel", VerticalScroll)
        w = Static(entry, classes="mod-entry")
        panel.mount(w)
        panel.scroll_end(animate=False)

    def _collect_triggers(
        self, seed: str, bios: dict[str, str]
    ) -> dict[str, list[str]]:
        """Ask each AI model for interrupt triggers. Returns {model_id: [trigger, ...]}."""
        ai_models = [m for m in self._models if m != HUMAN]

        self._setup_status = "🎯 collecting triggers…"
        self.call_from_thread(self._refresh_status)

        # Log header to moderator panel
        header = Text()
        header.append("── collecting triggers ──", style="dim bold")
        self.call_from_thread(self._log_to_moderator_panel, header)

        trigger_prompt = (
            f"Here's a conversation topic:\n\n{seed}\n\n"
            "List 3-5 short phrases describing moments where you'd be "
            "eager to jump in — topics that excite you, areas where you have something "
            "interesting to add, things that spark your curiosity or where you just "
            "can't help but contribute. "
            "Keep each phrase under 8 words. One per line, nothing else."
        )

        triggers: dict[str, list[str]] = {}
        lock = threading.Lock()

        def fetch_triggers(model: str) -> None:
            name, color = self._speaker_for(model)
            try:
                trigger_messages: list[ChatCompletionMessageParam] = [
                    {"role": "user", "content": trigger_prompt},
                ]
                _log_api_request(
                    "triggers", model, messages=trigger_messages, timeout=30.0
                )
                resp = self._clients[model].chat.completions.create(
                    model=model,
                    messages=trigger_messages,
                    timeout=30.0,
                )
                raw = (resp.choices[0].message.content or "").strip()
                # Parse lines, strip bullets/numbers
                lines = []
                for line in raw.splitlines():
                    line = line.strip()
                    line = line.lstrip("0123456789.-•*) ").strip()
                    if line:
                        lines.append(line)
                parsed = lines[:5]
                with lock:
                    triggers[model] = parsed
                # Log to moderator panel
                entry = Text()
                entry.append(f"[{name}] ", style=f"bold {color}")
                if parsed:
                    entry.append(" · ".join(parsed), style="dim")
                else:
                    entry.append("(none)", style="dim italic")
                self.call_from_thread(self._log_to_moderator_panel, entry)
            except Exception as exc:
                logger.warning("Failed to collect triggers from %s: %s", name, exc)
                with lock:
                    triggers[model] = []
                entry = Text()
                entry.append(f"[{name}] ", style=f"bold {color}")
                entry.append(f"error: {exc}", style="dim red")
                self.call_from_thread(self._log_to_moderator_panel, entry)

        threads = [
            threading.Thread(target=fetch_triggers, args=(m,), daemon=True)
            for m in ai_models
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Update bio widgets to show triggers (only if intros were collected)
        if self._intros_enabled:
            for i, model in enumerate(ai_models):
                model_triggers = triggers.get(model, [])
                if model_triggers:
                    name, color = self._speaker_for(model)
                    bio_text = bios.get(model, "")
                    trigger_line = " · ".join(model_triggers)
                    t = _prefixed_text(name, color, bio_text)
                    t.append(
                        f"\n  ↳ interrupts when: {trigger_line}", style="dim italic"
                    )
                    self.call_from_thread(self._update_message, f"bio-{i}", t)

        self._setup_status = None
        self.call_from_thread(self._refresh_status)

        return triggers

    def _handle_human_turn(
        self,
        worker: Worker,
        model: str,
        widget_id: str,
    ) -> str:
        """Handle a human participant's turn."""
        name, color = self._speaker_for(model)
        waiting_widget = Static(
            _prefixed_text(name, color, "waiting for input…", body_style="dim italic"),
            classes="message",
            id=widget_id,
        )
        waiting_widget.styles.border_left = ("tall", _css_color(color))
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
            self._history.append(Turn(speaker=model, content=f"[{name}]: (skipped)"))
            return model

        response_text = self._human_text
        self._update_prefixed(widget_id, name, color, response_text)
        self._history.append(Turn(speaker=model, content=f"[{name}]: {response_text}"))
        return model

    def _handle_ai_turn(
        self,
        model: str,
        widget_id: str,
        system_tmpl: str,
    ) -> str:
        """Handle an AI model's turn with streaming response."""
        name, color = self._speaker_for(model)
        # Build per-model bio block: other models' bios only (not this model's own)
        other_bios = {m: bio for m, bio in self._bios.items() if m != model and bio}
        if other_bios:
            bio_lines = "\n".join(
                f"- {short_name(m)}: {bio}" for m, bio in other_bios.items()
            )
            bio_block = (
                f"\n\nHere are the other participants' introductions:\n{bio_lines}"
            )
        else:
            bio_block = ""
        system_msg = system_tmpl.format(name=name) + bio_block

        other_names = [short_name(m) for m in self._models if m != model]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "request_next_speaker",
                    "description": (
                        "Request a specific participant to speak next. "
                        "Use this when you want to direct the conversation to someone, "
                        "e.g. to ask them a question or hear their perspective."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "participant": {
                                "type": "string",
                                "enum": other_names,
                                "description": "The name of the participant to speak next.",
                            }
                        },
                        "required": ["participant"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "skip",
                    "description": "Pass on your turn. Use this when you have nothing to add right now.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        full_messages: list[dict[str, object]] = [
            {"role": "system", "content": system_msg}
        ]
        for turn in self._history:
            role = "assistant" if turn.speaker == model else "user"
            msg: dict[str, object] = {"role": role, "content": turn.content}
            if role == "user" and turn.speaker is not None:
                msg["name"] = _api_name(short_name(turn.speaker))
            full_messages.append(msg)

        # Merge consecutive same-role messages (only when same speaker)
        merged: list[dict[str, object]] = []
        for msg in full_messages:
            if (
                merged
                and merged[-1]["role"] == msg["role"]
                and merged[-1].get("name") == msg.get("name")
            ):
                merged[-1]["content"] = (
                    str(merged[-1]["content"]) + "\n\n" + str(msg["content"])
                )
            else:
                merged.append(dict(msg))

        # Append word-limit directive
        if self._max_words is not None:
            word_directive = (
                f"\n\nSYSTEM DIRECTIVE: RESPOND IN {self._max_words} WORDS OR LESS"
            )
        else:
            word_directive = (
                "\n\nSYSTEM DIRECTIVE: No max word limit is currently enforced."
            )
        merged[-1]["content"] = str(merged[-1]["content"]) + word_directive

        # Some providers reject conversations ending with an assistant message
        if merged and merged[-1]["role"] == "assistant":
            merged.append({"role": "user", "content": "It's your turn to speak."})

        thinking_widget = Static(
            _prefixed_text(name, color, "thinking…", body_style="dim italic"),
            classes="message",
            id=widget_id,
        )
        thinking_widget.styles.border_left = ("tall", _css_color(color))
        self.call_from_thread(self._mount_message, thinking_widget)

        rendered_text, was_interrupted, tool_calls = self._stream(
            model, cast(list[ChatCompletionMessageParam], merged), widget_id, tools
        )

        # Process tool calls
        skipped = False
        for tc in tool_calls:
            if tc["name"] == "request_next_speaker":
                try:
                    args = json.loads(tc["arguments"])
                except (json.JSONDecodeError, KeyError):
                    continue
                participant = args.get("participant", "")
                for m in self._models:
                    if short_name(m) == participant:
                        self._next_override = m
                        break
            elif tc["name"] == "skip":
                skipped = True

        interrupter = self._model_interrupted_by
        self._model_interrupted_by = None

        if skipped and not rendered_text:
            self._update_prefixed(widget_id, name, color, "(passed)", body_style="dim")
            self._history.append(Turn(speaker=model, content=f"[{name}]: (passed)"))
        elif rendered_text:
            suffix = "—" if was_interrupted else ""
            interrupt_note = ""
            if was_interrupted and interrupter:
                interrupt_note = f" [interrupted by {interrupter}]"
            self._history.append(
                Turn(
                    speaker=model,
                    content=f"[{name}]: {rendered_text}{suffix}{interrupt_note}",
                )
            )
            self._update_prefixed(widget_id, name, color, rendered_text, suffix=suffix)
        elif was_interrupted:
            interrupt_note = ""
            if interrupter:
                interrupt_note = f" [interrupted by {interrupter}]"
            self._update_prefixed(
                widget_id, name, color, "(interrupted)", body_style="dim"
            )
            self._history.append(
                Turn(
                    speaker=model,
                    content=f"[{name}]: (interrupted){interrupt_note}",
                )
            )

        return model

    def _stream(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        widget_id: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, bool, list[dict]]:
        """Stream a model response with speed-controlled playback.

        Returns ``(rendered_text, was_interrupted, tool_calls)``.
        """
        name, color = self._speaker_for(model)

        token_buf: queue.Queue[str] = queue.Queue()
        stream_done = threading.Event()
        state = SimpleNamespace(error=None, tool_calls={}, rendered_text="")

        def producer() -> None:
            try:
                create_kwargs: dict = dict(
                    model=model,
                    messages=messages,
                    stream=True,
                    timeout=120.0,
                )
                if tools:
                    create_kwargs["tools"] = tools
                _log_api_request("stream", **create_kwargs)
                stream = self._clients[model].chat.completions.create(**create_kwargs)
                for chunk in stream:
                    if self._interrupted.is_set():
                        stream.close()
                        break
                    if chunk.choices and chunk.choices[0].delta.content:
                        token_buf.put(chunk.choices[0].delta.content)
                    if chunk.choices and chunk.choices[0].delta.tool_calls:
                        for tc in chunk.choices[0].delta.tool_calls:
                            idx = tc.index
                            if idx not in state.tool_calls:
                                state.tool_calls[idx] = {"name": "", "arguments": ""}
                            if tc.function:
                                if tc.function.name:
                                    state.tool_calls[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    state.tool_calls[idx]["arguments"] += (
                                        tc.function.arguments
                                    )
            except Exception as exc:
                state.error = exc
            finally:
                stream_done.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # ── Interrupt monitor thread ──────────────────────────────
        mod_client = self._moderator_client
        mod_model = self._moderator_model

        def monitor() -> None:
            """Watch rendered text for interrupt-trigger matches via the moderator model."""
            assert mod_client is not None and mod_model is not None
            other_triggers = {
                m: trigs
                for m, trigs in self._interrupt_triggers.items()
                if m != model and trigs
            }
            if not other_triggers:
                return

            trigger_list = "\n".join(
                f"- {short_name(m)}: {', '.join(trigs)}"
                for m, trigs in other_triggers.items()
            )

            last_checked_len = 0
            min_new_chars = 80  # wait for enough new content
            min_interval = 1.5  # seconds between checks
            last_check_time = 0.0

            # Log session header
            header = Text()
            header.append(f"── {name} speaking ──", style="dim bold")
            self.call_from_thread(self._log_to_moderator_panel, header)

            while not stream_done.is_set() and not self._interrupted.is_set():
                time.sleep(0.2)
                text = state.rendered_text
                new_len = len(text) - last_checked_len

                if new_len < min_new_chars:
                    continue

                now_mono = time.monotonic()
                if now_mono - last_check_time < min_interval:
                    continue

                # Look for sentence boundary in new content
                new_chunk = text[last_checked_len:]
                has_boundary = any(c in new_chunk for c in ".!?\n")
                if not has_boundary and new_len < 200:
                    continue

                last_checked_len = len(text)
                last_check_time = now_mono

                # Use last ~300 chars for context (keep prompt small)
                context = text[-300:] if len(text) > 300 else text
                # Snippet for the log (last ~60 chars)
                snippet = context[-60:].replace("\n", " ")
                if len(context) > 60:
                    snippet = "…" + snippet

                monitor_prompt = (
                    f"You are monitoring a group conversation for interrupt triggers.\n"
                    f"Current speaker: {name}\n\n"
                    f"Interrupt triggers for other participants:\n{trigger_list}\n\n"
                    f'Recent text from {name}:\n"{context}"\n\n'
                    f"Does the text clearly match any participant's interrupt trigger? "
                    f"Only match if it's strong and obvious. "
                    f"Reply with ONLY the participant name exactly as listed, or NONE."
                )

                try:
                    monitor_messages: list[ChatCompletionMessageParam] = [
                        {"role": "user", "content": monitor_prompt}
                    ]
                    _log_api_request(
                        "monitor",
                        mod_model,
                        messages=monitor_messages,
                        max_tokens=10,
                        temperature=0,
                        timeout=5.0,
                    )
                    resp = mod_client.chat.completions.create(
                        model=mod_model,
                        messages=monitor_messages,
                        max_tokens=10,
                        temperature=0,
                        timeout=5.0,
                    )
                    result = (resp.choices[0].message.content or "").strip()

                    # Log the check result
                    entry = Text()
                    entry.append(f'"{snippet}"\n', style="dim")
                    if result and result != "NONE":
                        entry.append(f"  → {result} ⚡", style="bold bright_yellow")
                    else:
                        entry.append("  → NONE", style="dim")
                    self.call_from_thread(self._log_to_moderator_panel, entry)

                    if result and result != "NONE":
                        # Match against known participant names
                        for m in other_triggers:
                            if short_name(m) == result:
                                if not stream_done.is_set():
                                    self._model_interrupted_by = short_name(m)
                                    self._next_override = m
                                    self._interrupted.set()
                                    with self._speed_cond:
                                        self._speed_cond.notify_all()
                                break
                except Exception as exc:
                    # Log errors too
                    err_entry = Text()
                    err_entry.append(f'"{snippet}"\n', style="dim")
                    err_entry.append(f"  → error: {exc}", style="dim red")
                    self.call_from_thread(self._log_to_moderator_panel, err_entry)

                if self._interrupted.is_set():
                    break

        monitor_thread: threading.Thread | None = None
        if mod_client and mod_model and self._interrupt_triggers:
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()

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
            state.rendered_text = rendered_text

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

        if was_interrupted:
            # Don't wait — daemon threads will clean up on their own
            return rendered_text, True, []

        producer_thread.join()
        if monitor_thread is not None:
            monitor_thread.join(timeout=6.0)

        parsed_tool_calls = list(state.tool_calls.values())

        if state.error and not rendered_text:
            self._update_prefixed(
                widget_id, name, color, f"Error: {state.error}", body_style="red"
            )
            return "", False, parsed_tool_calls

        if rendered_text:
            self._update_prefixed(
                widget_id,
                name,
                color,
                rendered_text,
                suffix="…" if was_interrupted else "",
            )
        elif not state.error and not parsed_tool_calls:
            self._update_prefixed(
                widget_id, name, color, "(empty response)", body_style="dim"
            )

        return rendered_text, was_interrupted, parsed_tool_calls


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

    # Parse optional moderator config
    moderator_model: str | None = None
    moderator_client: OpenAI | None = None
    mod_cfg = raw.get("moderator")
    if mod_cfg and isinstance(mod_cfg, dict):
        mod_url = mod_cfg.get("url")
        mod_token = mod_cfg.get("token", "")
        mod_model = mod_cfg.get("model", "")
        if not mod_url or not mod_token or not mod_model:
            print("Moderator config requires 'url', 'token', and 'model'.")
            sys.exit(1)
        moderator_model = mod_model
        moderator_client = OpenAI(base_url=mod_url, api_key=mod_token)

    intros_enabled = raw.get("intros", True)
    interrupts_enabled = raw.get("interrupts", True)

    app = TeaParty(
        AppConfig(
            models=all_models,
            clients=clients,
            moderator=moderator_model,
            moderator_client=moderator_client,
            intros=intros_enabled,
            interrupts=interrupts_enabled,
        )
    )
    try:
        app.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
