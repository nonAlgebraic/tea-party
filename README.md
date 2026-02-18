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

  "providers": [
    {
      "url": "https://openrouter.ai/api/v1",
      "token": "sk-or-...",
      "models": [
        "openai/gpt-5-pro",
        "anthropic/claude-opus-4.6",
        "google/gemini-3-pro-preview"
      ]
    },
    {
      "url": "https://api.openai.com/v1",
      "token": "sk-...",
      "models": [
        "gpt-4o"
      ]
    }
  ],

  // Optional: a fast model that monitors for interrupt triggers
  "moderator": {
    "url": "https://openrouter.ai/api/v1",
    "token": "sk-or-...",
    "model": "meta-llama/llama-3.1-8b-instruct"
  },

  // Both default to true
  "intros": true,
  "interrupts": true
}
```

Run:

```sh
python tea_party.py
# or, since the file has a shebang:
chmod +x tea_party.py
./tea_party.py
```

The `providers` array is **required** and must contain at least one provider with at least one model. Model IDs must be unique across all providers. The human participant (`you`) is always appended automatically — don't add `"human"` to any model list.

## Files

| File | Purpose |
|---|---|
| `tea_party.py` | The entire application — single file |
| `config.json` / `config.json5` | User configuration (providers, models, moderator) |
| `config.schema.json` | JSON Schema for the config file (draft 2020-12) |
| `tea_party.log` | API request log (recreated each run) |

## Architecture

### Dependencies

| Library | Role |
|---|---|
| **Textual** | TUI framework — layout, widgets, key bindings, event loop. Built on Rich. |
| **Rich** | Text formatting (`rich.text.Text`) for colored, styled message content. |
| **OpenAI SDK** | API client pointed at any OpenAI-compatible endpoint. Handles HTTP, SSE streaming, and response parsing. |
| **json5** | Config file parsing. Supports comments and trailing commas. |

### Multi-Provider Config

Rather than a single API token and endpoint, the config uses a `providers` array. Each provider has its own `url`, `token`, and `models` list. The app creates a separate `OpenAI` client per provider and maps each model ID to its client. This lets you mix models from OpenRouter, OpenAI direct, local servers, etc. in the same conversation.

### TUI Layout

Textual manages these docked regions:

```
┌────────────────────────────────────┬────────────────────┐
│  VerticalScroll #chat              │  #moderator-panel   │
│    Static .model-list              │  (hidden by default)│
│    Vertical #bio-container         │  Mod trigger checks │
│    Static .message (one per turn)  │  and decisions       │
│    ...                             │                     │
├────────────────────────────────────┴────────────────────┤
│  ChatInput (multi-line TextArea, when your turn)        │
├─────────────────────────────────────────────────────────┤
│  StatusBar #status (1 line)                             │
├─────────────────────────────────────────────────────────┤
│  Footer — key binding hints                             │
└─────────────────────────────────────────────────────────┘
```

On startup, a `ChatInput #seed-input` is shown for the seed prompt. It's removed after submission and the pre-conversation phases (bios, triggers) begin.

`ChatInput` is a `TextArea` subclass that supports multi-line editing with soft wrapping, auto-grows vertically (up to 14 lines), and submits on Enter.

### Threading Model

Textual's main thread owns the UI. The conversation loop runs in a **Textual thread worker** (`@work(thread=True)` on `_run_conversation`). All UI mutations from the worker go through `self.call_from_thread(...)`, which is Textual's thread-safe bridge.

There are three `threading.Event` objects for cross-thread coordination:

| Event | Purpose |
|---|---|
| `_pause_gate` | Starts **cleared** (hold mode). Set when running. The worker blocks on `_pause_gate.wait()` between turns. |
| `_interrupted` | Set when the user presses Enter during a turn, or when the moderator triggers an interrupt. Checked by both the consumer loop and the API producer thread. |
| `_human_ready` | Set when the human submits input. The worker blocks on this during human turns. |

A `threading.Condition` (`_speed_cond`) coordinates speed changes and interrupt wakeups for the consumer loop.

### Pre-Conversation Phases

Before the main conversation loop, up to two optional setup phases run:

**1. Introductions** (`_collect_bios`, enabled by `"intros": true`)

Each AI model streams a short personality bio in parallel. The human also enters a bio via `ChatInput`. Results are displayed in a bordered "Introductions" container. During the conversation, each model's system prompt includes other participants' bios (but not its own).

**2. Interrupt Triggers** (`_collect_triggers`, enabled by `"interrupts": true`, requires a moderator)

Each AI model provides 3–5 short phrases describing moments where it would want to jump in. These are collected in parallel (non-streaming) and logged to the moderator panel. If intros are also enabled, triggers are shown appended to each model's bio widget.

### Conversation Loop (`_run_conversation`)

Runs in the worker thread. Infinite loop:

1. **Hold check** — blocks on `_pause_gate`. The app starts in hold mode; the user must press Enter or Ctrl+P to begin.
2. **Single-step** — if `_advance_once` is set (from pressing Enter while held), the gate is re-closed after one turn.
3. **Pick model** — uses `_next_override` if set (from Ctrl+R cycling or tool calls), otherwise random (excluding whoever spoke last).
4. **Human turn** — if the picked participant is `HUMAN`: mount a `ChatInput` widget, block on `_human_ready`, record the text.
5. **AI turn** — build the message list, mount a "thinking…" widget, call `_stream()`.
6. **Record history** — append the rendered text (what was actually displayed, not what the API generated) to `history`.

### Message Perspective

Each AI model sees the conversation from its own point of view:

- Its own previous messages → `role: "assistant"`
- Everyone else's messages (other AIs, the human, the seed) → `role: "user"` with a `name` field
- A `system` message telling it who it is, who the other participants are, behavioral guidance, and other participants' bios

Consecutive same-role messages from the same speaker are merged (concatenated with `\n\n`) to satisfy providers that reject adjacent messages with the same role. If the conversation ends with an assistant message, a synthetic user message is appended.

A word-limit directive is appended to the last message in the conversation, enforcing the current `_max_words` setting.

### Tools

Each AI model has two tools available:

| Tool | Effect |
|---|---|
| `request_next_speaker` | Sets `_next_override` so a specific participant speaks next. Useful when a model asks someone a direct question. |
| `skip` | Passes the turn. The message shows "(passed)" instead of content. |

Tool calls are extracted from the streaming response and processed after the stream completes.

### The System Prompt

Each model gets a tailored system prompt that tells it:

- Its own identity ("You are gpt-5-pro")
- To be concise and conversational
- NOT to prefix responses with its name (the UI does that)
- That interrupted messages are normal (cut off by the moderator)
- How to use `request_next_speaker` and `skip` tools
- Other participants' bios (excluding its own)

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

On interrupt (user or moderator), the consumer stops immediately and discards any tokens still in the queue. Only `rendered_text` — what was actually displayed — goes into history. The history records an em dash suffix (`—`) and, if the moderator triggered it, a note of who interrupted.

**Speed scale** (controlled by `_tps`):

| Value | Meaning |
|---|---|
| `None` | No limit — tokens render as fast as they arrive |
| `0` | Frozen — tokens buffer but nothing renders until speed is increased |
| `3–500` | Controlled rate in tokens/second |

Speed control keys are currently disabled in the UI; `_tps` defaults to `None` (unlimited).

### Moderator & Interrupt Monitor

When a moderator model is configured and interrupt triggers have been collected, a **monitor thread** runs alongside each AI turn's stream. It periodically checks the rendered text against other participants' triggers by sending a short prompt to the moderator model:

- Waits for at least 80 new characters and 1.5 seconds between checks
- Looks for sentence boundaries before checking (to avoid partial-sentence false positives)
- Sends the last ~300 characters of rendered text plus a list of all other participants' triggers
- The moderator replies with either a participant name or "NONE"
- If a participant is named, the monitor interrupts the current speaker and sets that participant as next speaker

All moderator activity is logged to the moderator side panel (toggle with Ctrl+M).

### Rewind Mode

Pressing Ctrl+Z enters rewind mode (only available when no one is speaking and at least one message exists). In rewind mode:

- The conversation is paused
- Up/Down arrows navigate through message history, highlighting the selected message
- Enter commits: all messages after the selected point are removed from both the UI and history
- Escape cancels and restores the previous pause state

After committing a rewind, the conversation resumes from the truncated point.

### Key Bindings

| Key | Action | Notes |
|---|---|---|
| `enter` | Pass the mic | Interrupts current speaker, or advances one turn if held |
| `ctrl+p` | Hold / Go | Toggles between held (paused between turns) and running |
| `ctrl+r` | Cycle next speaker | Cycles through models, then back to random |
| `ctrl+z` | Rewind | Enter/exit rewind mode |
| `]` | More words | Increases max word limit (10→25→50→100→200→500→unlimited) |
| `[` | Fewer words | Decreases max word limit |
| `ctrl+m` | Moderator panel | Toggles the side panel showing moderator activity |
| `ctrl+q` | Quit | |

Enter and the word-limit keys are handled in `on_key` (not as `Binding`s) so they can be context-sensitive — they yield to `ChatInput` when one is focused and have special behavior in rewind mode.

### Status Bar

A 1-line `StatusBar` widget docked to the bottom:

State (`✏️ enter seed prompt` / `🪪 generating bios…` / `🎯 collecting triggers…` / `💬 model-name` / `▶ running`) │ Next (`🎲 next → random` / `🎯 next → model-name` / `⏪ rewind`) │ Turn count with hold indicator (`turn N ✋` or `turn N ▶`) │ Word limit (`📏 unlimited` / `📏 ≤50w`)

### Model Colors

Each model gets a color from `MODEL_COLORS` (indexed by position, wraps with modulo over 10 entries). Colors are used in:

- The participant list on startup
- Message prefixes (`[model-name]`)
- Left border on message widgets
- The status bar's "next →" indicator

### Error Handling

- API errors (non-200, network failures, timeouts) are caught in `_stream` and displayed inline as a red error message in the chat. The conversation continues — it doesn't crash.
- `action_quit_app` unblocks all waiting threads (`_interrupted`, `_pause_gate`, `_human_ready`) before calling `self.exit()` so the worker thread can terminate cleanly without tracebacks.
- Missing config or providers is caught at startup with a clear message.

### Config Loading

`_load_config()` checks for `config.json5` first, then `config.json`, in the same directory as the script. Parsed with `json5.load()` (which also handles standard JSON). If neither file exists, it falls back to an empty dict and the missing-providers check fails with a message.

The `HUMAN` entry is always appended as the last participant after all provider models are loaded.

### API Logging

Every API request (bios, triggers, monitor checks, conversation streams) is logged to `tea_party.log` in the application directory. The log is recreated on each run and includes timestamps and full request bodies formatted as JSON.