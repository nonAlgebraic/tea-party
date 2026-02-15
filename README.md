# Tea Party

A terminal application that drops multiple LLMs into a group conversation and lets them talk. You provide a seed prompt, and models take turns responding — chosen at random — until you stop them. You're also a participant.

## Quick Start

```sh
cd tea-party
python3 -m venv .venv
source .venv/bin/activate
pip install textual rich openai json5
```

Create `config.json` (or `config.json5` — JSON5 is supported for comments):

```json5
{
  "$schema": "./config.schema.json",

  // Or set OPENROUTER_API_KEY env var instead
  "apiToken": "sk-or-...",

  "models": [
    "openai/gpt-5-pro",
    "anthropic/claude-opus-4.6",
    "google/gemini-3-pro-preview",
    "meta-llama/llama-4-maverick",
    "x-ai/grok-4"
  ]
}
```

Run:

```sh
python tea_party.py
# or, since the file has a shebang:
chmod +x tea_party.py
./tea_party.py
```

The `models` array is **required** and must contain at least 2 model IDs. `apiToken` is optional if you set `OPENROUTER_API_KEY` in your environment (config takes priority if both exist). The human participant (`you`) is always appended automatically — don't add `"human"` to the models list.

## Files

| File | Purpose |
|---|---|
| `tea_party.py` | The entire application — single file |
| `config.json` / `config.json5` | User configuration (models, API token) |
| `config.schema.json` | JSON Schema for the config file (draft 2020-12) |

## Architecture

### Dependencies

| Library | Role |
|---|---|
| **Textual** | TUI framework — layout, widgets, key bindings, event loop. Built on Rich. |
| **Rich** | Text formatting (`rich.text.Text`) for colored, styled message content. |
| **OpenAI SDK** | API client pointed at OpenRouter (`base_url="https://openrouter.ai/api/v1"`). Handles HTTP, SSE streaming, and response parsing. |
| **json5** | Config file parsing. Supports comments and trailing commas. |

No hand-rolled ANSI escape codes, SSE parsers, or terminal manipulation. Every "hard" problem is delegated to a battle-tested library.

### TUI Layout

Textual manages four docked regions:

```
┌──────────────────────────────────────┐
│  VerticalScroll #chat                │  ← scrollable conversation history
│    Static .message (one per turn)    │     each turn is a separate widget
│    Static .message                   │
│    ...                               │
├──────────────────────────────────────┤
│  Input #human-input (when your turn) │  ← only mounted during human turns
├──────────────────────────────────────┤
│  StatusBar #status (2 lines)         │  ← persistent state display
├──────────────────────────────────────┤
│  Footer                             │  ← key binding hints (from Textual)
└──────────────────────────────────────┘
```

On startup, an `Input #seed-input` is shown for the seed prompt. It's removed after submission and the conversation begins.

### Threading Model

Textual's main thread owns the UI. The conversation loop runs in a **Textual thread worker** (`@work(thread=True)` on `_run_conversation`). All UI mutations from the worker go through `self.call_from_thread(...)`, which is Textual's thread-safe bridge.

There are three `threading.Event` objects for cross-thread coordination:

| Event | Purpose |
|---|---|
| `_pause_gate` | Normally **set** (open). Cleared when paused. The worker blocks on `_pause_gate.wait()`. |
| `_interrupted` | Set when the user presses ctrl+n. Checked by both the consumer loop and the API producer thread. |
| `_human_ready` | Set when the human submits input. The worker blocks on this during human turns. |

### Conversation Loop (`_run_conversation`)

Runs in the worker thread. Infinite loop:

1. **Pause check** — blocks on `_pause_gate` if paused.
2. **Pick model** — uses `_next_override` if set (from number keys), otherwise random (excluding whoever spoke last).
3. **Human turn** — if the picked participant is `HUMAN`: mount an `Input` widget, block on `_human_ready`, record the text.
4. **AI turn** — build the message list, mount a "thinking…" widget, call `_stream()`.
5. **Record history** — append the rendered text (what was actually displayed, not what the API generated) to `history`.

### Message Perspective

Each AI model sees the conversation from its own point of view:

- Its own previous messages → `role: "assistant"`
- Everyone else's messages (other AIs, the human, the seed) → `role: "user"`
- A `system` message telling it who it is, who the other participants are, and behavioral guidance

Consecutive same-role messages are merged (concatenated with `\n\n`) to satisfy providers that reject adjacent messages with the same role.

### The System Prompt

Each model gets a tailored system prompt (filled from `system_tmpl`) that tells it:

- Its own identity ("You are gpt-5-pro")
- Who the other participants are (including that "you" is a human)
- To be concise and conversational
- NOT to prefix responses with its name (the UI does that)
- That it can pass its turn
- That interruptions happen and truncated messages are normal

### Token Buffer & Speed Control (`_stream`)

Streaming uses a **producer/consumer** pattern to decouple network speed from display speed:

```
  API (full speed)          Queue             Screen (controlled speed)
  ┌──────────┐         ┌──────────┐          ┌──────────┐
  │ Producer │ ──put──▶│ token_buf│ ──get──▶  │ Consumer │
  │ (thread) │         │ (Queue)  │           │ (worker) │
  └──────────┘         └──────────┘          └──────────┘
```

**Producer** — a daemon thread that calls `client.chat.completions.create(stream=True)` and pushes every token into a `queue.Queue`. Runs at whatever speed the API delivers. Stops on interrupt or stream end.

**Consumer** — the worker thread pops tokens from the queue and renders them. Between each token, it sleeps for `1.0 / self._tps` seconds (unless unlimited). UI updates are throttled to ~30fps regardless of token rate.

**Why this matters for interrupts**: when the user interrupts (ctrl+n), the consumer stops immediately. Tokens still in the queue are **discarded**. Only `rendered_text` — what was actually displayed on screen — goes into the conversation history. Other models see exactly what the human saw, nothing more. The interruption is real, not cosmetic. The display appends `…` (ellipsis) to the truncated message, while the history records `—` (em dash) so other models can recognize the cut-off.

**Speed scale** (controlled by `_tps`):

| Value | Meaning |
|---|---|
| `None` | No limit — tokens render as fast as they arrive |
| `0` | Frozen — tokens buffer but nothing renders until speed is increased |
| `3–500` | Controlled rate in tokens/second |

The `[` key steps down (÷1.5, bottoms out at 0), `]` steps up (×1.5, tops out then jumps to unlimited), `\` jumps directly to unlimited. Speed changes take effect mid-stream because `_tps` is read on every token.

### Key Bindings

All bindings are **non-priority**, meaning they naturally yield to the `Input` widget when one is focused (seed prompt, human turn). When no input is mounted, they fire at the app level.

| Key | Action | Notes |
|---|---|---|
| `space` | Pause / resume | Pauses between turns, not mid-token |
| `ctrl+n` | Interrupt current speaker | Discards buffered tokens, next model speaks |
| `ctrl+r` | Clear next-model override | Resets to random selection |
| `\` | Set speed to unlimited | |
| `]` | Increase speed (×1.5) | |
| `[` | Decrease speed (÷1.5) | Goes to 0 (frozen) below minimum |
| `1`–`9` | Pick next speaker | One-shot — resets to random after that turn |
| `q` | Quit | Ignored when an Input is focused |

Number keys and speed keys are handled in `on_key` (not as `Binding`s) so they don't clutter the footer and can be guarded against firing during human input.

### Status Bar

A 2-line `StatusBar` widget (subclass of `Static`, docked to bottom):

**Line 1**: State (`▶ running` / `⏸ PAUSED` / `💬 model-name`) │ Next (`🔄 auto` / `🎯 next → model-name`) │ Turn count │ Speed (`⚡ unlimited` / `⚡ frozen` / `⚡ N tok/s`)

**Line 2**: Numbered model key mapping in each model's color (`1=gpt-5-pro  2=claude-opus-4.6  ...`)

Updated via `_refresh_status()`, which is called from both the main thread (key handlers) and the worker thread (via `call_from_thread`).

### Model Colors

Each model gets a color from `MODEL_COLORS` (indexed by position in `MODELS`). The palette has 10 entries and wraps with modulo. Colors are used in:

- The participant list on startup
- Message prefixes (`[model-name]`)
- The status bar model key mapping
- The "next →" indicator

### Error Handling

- API errors (non-200, network failures, timeouts) are caught in `_stream` and displayed inline as a red error message in the chat. The conversation continues — it doesn't crash.
- `action_quit_app` unblocks all waiting threads (`_interrupted`, `_pause_gate`, `_human_ready`) before calling `self.exit()` so the worker thread can terminate cleanly without tracebacks.
- Missing config or API key is caught at startup with a clear message.

### Config Loading

`_load_config()` checks for `config.json5` first, then `config.json`, in the same directory as the script. Parsed with `json5.load()` (which also handles standard JSON). If neither file exists, it falls back to an empty dict and the missing-models check fails with a message.

The `"human"` entry is stripped from the models list if present, then `HUMAN` is always appended as the last participant. This guarantees the human is always in the conversation regardless of config.