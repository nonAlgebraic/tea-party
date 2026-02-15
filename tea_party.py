#!/usr/bin/env python3

import os
import sys
import random
import threading
from openai import OpenAI
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static
from textual.worker import get_current_worker

# ── Models ────────────────────────────────────────────────────────────────────
HUMAN = "human"

MODELS = [
    "openai/gpt-5.2-pro",
    "anthropic/claude-opus-4.6",
    "google/gemini-3-pro-preview",
    "z-ai/glm-5",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-large-2512",
    "x-ai/grok-4",
    HUMAN,
]

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

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def short_name(model_id: str) -> str:
    if model_id == HUMAN:
        return "you"
    return model_id.split("/")[-1]


def color_for(model_id: str) -> str:
    idx = MODELS.index(model_id) if model_id in MODELS else 0
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def _build_message_text(name: str, color: str, body: str, suffix: str = "") -> Text:
    t = Text()
    t.append(f"[{name}] ", style=f"bold {color}")
    t.append(body + suffix)
    return t


def _thinking_text(name: str, color: str) -> Text:
    t = Text()
    t.append(f"[{name}] ", style=f"bold {color}")
    t.append("thinking…", style="dim italic")
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
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self):
        super().__init__()
        self._conversation_started = False
        self._is_paused = False
        self._speaking: str | None = None
        self._next_override: str | None = None
        self._turn = 0
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
        if not OPENROUTER_API_KEY:
            chat = self.query_one("#chat")
            chat.mount(Static("[bold red]Set OPENROUTER_API_KEY environment variable.[/]"))
            self.query_one("#seed-input").remove()
            return

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
        try:
            self.query_one("#human-input")
            return
        except Exception:
            pass
        if event.character and event.character in "123456789":
            idx = int(event.character) - 1
            if idx < len(MODELS):
                self._next_override = MODELS[idx]
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

    def _refresh_status(self) -> None:
        # Line 1: state
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

        line1 = " │ ".join(state_parts)

        # Line 2: model keys
        model_keys = "  ".join(
            f"[{color_for(m)}]{i+1}={short_name(m)}[/{color_for(m)}]"
            for i, m in enumerate(MODELS)
        )
        line2 = model_keys

        try:
            self.query_one("#status", StatusBar).update(f"{line1}\n{line2}")
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

    def _show_human_input(self) -> None:
        inp = Input(placeholder="Your turn — type your message…", id="human-input")
        self.mount(inp, after=self.query_one("#status"))
        inp.focus()

    def _hide_human_input(self) -> None:
        try:
            self.query_one("#human-input").remove()
        except Exception:
            pass

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
            while not self._pause_gate.wait(timeout=0.3):
                if worker.is_cancelled:
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

            name = short_name(model)
            color = color_for(model)
            self._speaking = name
            self.call_from_thread(self._refresh_status)

            widget_id = f"msg-{self._turn}"

            if model == HUMAN:
                # Human's turn — show input and wait
                waiting_text = Text()
                waiting_text.append(f"[{name}] ", style=f"bold {color}")
                waiting_text.append("waiting for input…", style="dim italic")
                waiting_widget = Static(waiting_text, classes="message", id=widget_id)
                self.call_from_thread(self._mount_message, waiting_widget)
                self.call_from_thread(self._show_human_input)

                self._human_ready.clear()
                while not self._human_ready.wait(timeout=0.3):
                    if worker.is_cancelled:
                        return
                    if self._interrupted.is_set():
                        # User interrupted their own turn — skip it
                        self.call_from_thread(self._hide_human_input)
                        self.call_from_thread(
                            self._update_message, widget_id,
                            _build_message_text(name, color, "(skipped)", ""),
                        )
                        self._speaking = None
                        self.call_from_thread(self._refresh_status)
                        last_model = model
                        break
                else:
                    # Got human input
                    response_text = self._human_text
                    self.call_from_thread(
                        self._update_message, widget_id,
                        _build_message_text(name, color, response_text),
                    )
                    history.append((model, f"[{name}]: {response_text}"))
                    last_model = model
                    self._speaking = None
                    self.call_from_thread(self._refresh_status)
                continue

            # AI model's turn
            # Build messages from this model's perspective
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
            thinking_widget = Static(_thinking_text(name, color), classes="message", id=widget_id)
            self.call_from_thread(self._mount_message, thinking_widget)

            # Stream response
            response_text, was_interrupted = self._stream(model, merged, widget_id)

            if response_text:
                suffix = "—" if was_interrupted else ""
                history.append((model, f"[{name}]: {response_text}{suffix}"))
                self.call_from_thread(
                    self._update_message, widget_id,
                    _build_message_text(name, color, response_text, suffix),
                )
            else:
                # empty or error — widget already updated in _stream
                pass

            last_model = model
            self._speaking = None
            self.call_from_thread(self._refresh_status)

    def _stream(self, model: str, messages: list[dict], widget_id: str) -> tuple[str, bool]:
        name = short_name(model)
        color = color_for(model)
        full_text = ""
        was_interrupted = False

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                timeout=120.0,
            )

            for chunk in stream:
                if self._interrupted.is_set():
                    was_interrupted = True
                    stream.close()
                    break

                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_text += token
                    self.call_from_thread(
                        self._update_message, widget_id,
                        _build_message_text(name, color, full_text),
                    )

        except Exception as e:
            err_text = Text()
            err_text.append(f"[{name}] ", style=f"bold {color}")
            err_text.append(f"Error: {e}", style="red")
            self.call_from_thread(self._update_message, widget_id, err_text)
            return "", False

        if was_interrupted:
            self.call_from_thread(
                self._update_message, widget_id,
                _build_message_text(name, color, full_text, "…"),
            )
        elif not full_text:
            empty = Text()
            empty.append(f"[{name}] ", style=f"bold {color}")
            empty.append("(empty response)", style="dim")
            self.call_from_thread(self._update_message, widget_id, empty)

        return full_text, was_interrupted


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    app = TeaParty()
    app.run()
