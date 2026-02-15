#!/usr/bin/env python3

import os
import sys
import random
import threading
import queue
import time
from pathlib import Path
import json5
from openai import OpenAI
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static
from textual.worker import get_current_worker

# ── Config ────────────────────────────────────────────────────────────────────
HUMAN = "human"

CONFIG_DIR = Path(__file__).parent


def _load_config() -> dict:
    for name in ("config.json5", "config.json"):
        path = CONFIG_DIR / name
        if path.exists():
            with open(path) as f:
                return json5.load(f)
    return {}


_config = _load_config()

if "models" not in _config:
    print("config.json must contain a 'models' array.")
    sys.exit(1)

MODELS: list[str] = [m for m in _config["models"] if m != HUMAN] + [HUMAN]

OPENROUTER_API_KEY: str = (
    _config.get("apiToken")
    or os.environ.get("OPENROUTER_API_KEY")
    or ""
)

MODEL_COLORS = [
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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Speed presets: tokens per second (None = no limit, 0 = frozen)
SPEED_MIN = 3
SPEED_MAX = 500
SPEED_DEFAULT = None
SPEED_FACTOR = 1.5


def short_name(model_id: str) -> str:
    if model_id == HUMAN:
        return "you"
    return model_id.split("/")[-1]


def color_for(model_id: str) -> str:
    idx = MODELS.index(model_id) if model_id in MODELS else 0
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def _speaker(model_id: str) -> tuple[str, str]:
    return short_name(model_id), color_for(model_id)


def _prefixed_text(name: str, color: str, body: str, body_style: str = "", suffix: str = "") -> Text:
    t = Text()
    t.append(f"[{name}] ", style=f"bold {color}")
    t.append(body + suffix, style=body_style)
    return t


# ── App ───────────────────────────────────────────────────────────────────────

class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 2;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """


class TeaParty(App):
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
    #seed-input {
        dock: bottom;
    }
    #human-input {
        dock: bottom;
        border-top: solid $accent;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_pause", "Pause/Resume", show=True),
        Binding("ctrl+n", "interrupt", "Next Speaker", show=True),
        Binding("ctrl+r", "randomize_next", "Random Next", show=True),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(self):
        super().__init__()
        self._conversation_started = False
        self._is_paused = False
        self._speaking: str | None = None
        self._next_override: str | None = None
        self._turn = 0
        self._tps: float | None = SPEED_DEFAULT  # None = no limit, 0 = frozen
        self._interrupted = threading.Event()
        self._pause_gate = threading.Event()
        self._pause_gate.set()  # start unpaused (gate open)
        self._human_ready = threading.Event()
        self._human_text = ""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        yield StatusBar(id="status")
        yield Input(placeholder="Enter seed prompt…", id="seed-input")
        yield Footer()

    def on_mount(self) -> None:
        # Show model list in chat
        chat = self.query_one("#chat")
        lines = Text()
        lines.append("Participants:\n", style="bold")
        for i, m in enumerate(MODELS):
            c = color_for(m)
            lines.append(f"  {i + 1}. {short_name(m)}\n", style=c)
        lines.append("\nPress a number key during conversation to pick who speaks next.\n", style="dim")
        info = Static(lines, classes="model-list")
        chat.mount(info)

        self.query_one("#seed-input").focus()
        self._refresh_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "seed-input":
            seed = event.value.strip()
            if not seed:
                return
            self.query_one("#seed-input").remove()
            self._conversation_started = True
            self._refresh_status()
            self._run_conversation(seed)
        elif event.input.id == "human-input":
            text = event.value.strip()
            if not text:
                return
            self._human_text = text
            self._human_ready.set()
            event.input.remove()

    def on_key(self, event) -> None:
        if not self._conversation_started:
            return
        # Don't intercept keys when human input is active
        if self.query("#human-input"):
            return
        if event.character and event.character in "123456789":
            idx = int(event.character) - 1
            if idx < len(MODELS):
                self._next_override = MODELS[idx]
                self._refresh_status()
        elif event.character:
            speed_keys = {"]": 1, "[": -1, "\\": 0}
            if event.character in speed_keys:
                self._adjust_speed(speed_keys[event.character])

    def _adjust_speed(self, direction: int) -> None:
        """direction: +1 = faster, -1 = slower, 0 = unlimited"""
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
        self._refresh_status()

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
        # Also unpause if paused, so it moves on
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
        # Don't quit if human input or seed input is focused — q is a letter
        if self.query("#human-input") or self.query("#seed-input"):
            return
        # Unblock any waiting threads so they can see cancellation
        self._interrupted.set()
        self._pause_gate.set()
        self._human_ready.set()
        self.exit()

    def _refresh_status(self) -> None:
        # Line 1: state + speed
        state_parts: list[str] = []

        if self._is_paused:
            state_parts.append("⏸  PAUSED")
        elif self._speaking:
            state_parts.append(f"💬 {self._speaking}")
        else:
            state_parts.append("▶  running")

        if self._next_override:
            c = color_for(self._next_override)
            state_parts.append(f"🎯 next → [{c}]{short_name(self._next_override)}[/{c}]")
        else:
            state_parts.append("🔄 auto")

        state_parts.append(f"turn {self._turn}")

        # Speed display
        hint = "[dim]\\[slow \\]fast \\\\unlim[/dim]"
        if self._tps is None:
            speed = "unlimited"
        elif self._tps == 0:
            speed = "frozen"
        else:
            speed = f"{int(round(self._tps))} tok/s"
        state_parts.append(f"⚡ {speed} {hint}")

        line1 = " │ ".join(state_parts)

        # Line 2: model keys
        model_keys = "  ".join(
            f"[{c}]{i+1}={short_name(m)}[/{c}]"
            for i, m in enumerate(MODELS)
            for c in (color_for(m),)
        )

        try:
            self.query_one("#status", StatusBar).update(f"{line1}\n{model_keys}")
        except Exception:
            pass

    def _mount_message(self, widget: Static) -> None:
        chat = self.query_one("#chat")
        chat.mount(widget)
        chat.scroll_end(animate=False)

    def _update_message(self, widget_id: str, content: Text) -> None:
        try:
            w = self.query_one(f"#{widget_id}", Static)
            w.update(content)
            self.query_one("#chat").scroll_end(animate=False)
        except Exception:
            pass

    def _update_prefixed(self, widget_id: str, name: str, color: str, body: str, **kw) -> None:
        self.call_from_thread(self._update_message, widget_id, _prefixed_text(name, color, body, **kw))

    def _show_human_input(self) -> None:
        inp = Input(placeholder="Your turn — type your message…", id="human-input")
        self.mount(inp, after=self.query_one("#status"))
        inp.focus()

    def _hide_human_input(self) -> None:
        for w in self.query("#human-input"):
            w.remove()

    def _wait_or_cancel(self, event: threading.Event, worker, also_break_on: threading.Event | None = None) -> str:
        """Block until event is set. Returns 'ready', 'cancelled', or 'interrupted'."""
        while not event.wait(timeout=0.3):
            if worker.is_cancelled:
                return "cancelled"
            if also_break_on and also_break_on.is_set():
                return "interrupted"
        return "ready"

    @work(thread=True)
    def _run_conversation(self, seed: str) -> None:
        worker = get_current_worker()

        model_names = ", ".join(short_name(m) for m in MODELS)
        intro = (
            f"You are all in a group conversation together. "
            f"The participants are: {model_names}. "
            f"Most of you are AI models, but 'you' is a human participant. "
            f"A human has set up this room for everyone to chat. Here's the topic:\n\n{seed}"
        )
        history: list[tuple[str | None, str]] = [(None, intro)]
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
            # Pause gate — blocks while paused
            if self._wait_or_cancel(self._pause_gate, worker) == "cancelled":
                return

            self._interrupted.clear()
            self._turn += 1

            # Pick model
            override = self._next_override
            self._next_override = None

            if override and override != last_model:
                model = override
            else:
                candidates = [m for m in MODELS if m != last_model]
                model = random.choice(candidates)

            self._speaking = short_name(model)
            self.call_from_thread(self._refresh_status)

            widget_id = f"msg-{self._turn}"

            if model == HUMAN:
                last_model = self._handle_human_turn(worker, model, widget_id, history)
            else:
                last_model = self._handle_ai_turn(model, widget_id, history, system_tmpl)

            self._speaking = None
            self.call_from_thread(self._refresh_status)

    def _handle_human_turn(self, worker, model, widget_id, history) -> str:
        name, color = _speaker(model)
        waiting_widget = Static(
            _prefixed_text(name, color, "waiting for input…", body_style="dim italic"),
            classes="message", id=widget_id,
        )
        self.call_from_thread(self._mount_message, waiting_widget)
        self.call_from_thread(self._show_human_input)

        self._human_ready.clear()
        result = self._wait_or_cancel(self._human_ready, worker, also_break_on=self._interrupted)

        if result == "cancelled":
            return model
        elif result == "interrupted":
            self.call_from_thread(self._hide_human_input)
            self._update_prefixed(widget_id, name, color, "(skipped)")
            return model

        # Got human input
        response_text = self._human_text
        self._update_prefixed(widget_id, name, color, response_text)
        history.append((model, f"[{name}]: {response_text}"))
        return model

    def _handle_ai_turn(self, model, widget_id, history, system_tmpl) -> str:
        name, color = _speaker(model)
        others = ", ".join(short_name(m) for m in MODELS if m != model)
        system_msg = system_tmpl.format(name=name, others=others)

        full_messages: list[dict] = [{"role": "system", "content": system_msg}]
        for speaker, content in history:
            role = "assistant" if speaker == model else "user"
            full_messages.append({"role": role, "content": content})

        # Merge consecutive same-role messages
        merged: list[dict] = []
        for msg in full_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(dict(msg))

        # Mount thinking widget
        thinking_widget = Static(
            _prefixed_text(name, color, "thinking…", body_style="dim italic"),
            classes="message", id=widget_id,
        )
        self.call_from_thread(self._mount_message, thinking_widget)

        # Stream response with buffered playback
        rendered_text, was_interrupted = self._stream(model, merged, widget_id)

        if rendered_text:
            suffix = "—" if was_interrupted else ""
            history.append((model, f"[{name}]: {rendered_text}{suffix}"))
            self._update_prefixed(widget_id, name, color, rendered_text, suffix=suffix)

        return model

    def _stream(self, model: str, messages: list[dict], widget_id: str) -> tuple[str, bool]:
        name, color = _speaker(model)

        token_buf: queue.Queue[str] = queue.Queue()
        stream_done = threading.Event()
        stream_error: list[Exception | None] = [None]

        # ── Producer: fills buffer from API at full speed ─────────────
        def producer():
            try:
                stream = client.chat.completions.create(
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
            except Exception as e:
                stream_error[0] = e
            finally:
                stream_done.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # ── Consumer: renders from buffer at controlled speed ─────────
        rendered_text = ""
        was_interrupted = False
        last_render = 0.0
        MIN_RENDER_INTERVAL = 1.0 / 30  # cap UI updates at ~30fps

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

            # Throttle UI updates to avoid overwhelming Textual
            now = time.monotonic()
            if now - last_render >= MIN_RENDER_INTERVAL:
                self._update_prefixed(widget_id, name, color, rendered_text)
                last_render = now

            # Speed-controlled delay between tokens
            if self._tps is None:
                pass  # no limit
            elif self._tps == 0:
                # Frozen — wait until speed changes or interrupted
                while self._tps == 0 and not self._interrupted.is_set():
                    time.sleep(0.05)
            else:
                time.sleep(1.0 / self._tps)

        # Wait for producer to finish (it will stop on interrupt or naturally)
        producer_thread.join(timeout=2.0)

        # Handle errors
        if stream_error[0] and not rendered_text:
            self._update_prefixed(widget_id, name, color, f"Error: {stream_error[0]}", body_style="red")
            return "", False

        # Final render to make sure everything rendered is visible
        if rendered_text:
            self._update_prefixed(widget_id, name, color, rendered_text, suffix="…" if was_interrupted else "")
        elif not stream_error[0]:
            self._update_prefixed(widget_id, name, color, "(empty response)", body_style="dim")

        return rendered_text, was_interrupted


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Set apiToken in config.json or OPENROUTER_API_KEY env var.")
        sys.exit(1)
    app = TeaParty()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
