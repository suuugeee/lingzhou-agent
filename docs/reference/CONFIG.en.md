# Configuration Reference

[中文](CONFIG.md) | [English](CONFIG.en.md)

By default, configuration lives in `~/.lingzhou/lingzhou.json`. The runtime can inspect and update config through `config.get` and `config.set`, and then reload it without restarting the whole process.

The default runtime directory layout is under `~/.lingzhou/`: `db_path`, `memory_dir`, `state_dir`, `workspace_dir`, logs, and temporary artifacts should stay there in production instead of being written back into the source repository.

## Model Routing

```jsonc
{
  "model": "bailian/qwen3.6-plus",
  "vision_model": "copilot/gpt-5.4",
  "routing": {
    "reader": "bailian/qwen-plus",
    "reasoner": "copilot/gpt-5.4",
    "repair": "bailian/qwen3.6-plus"
  }
}
```

Providers are defined in the `providers` section. Environment variables or auth profiles remain the recommended source of API keys; the setup flow may also store a key directly in a local machine config, but that file should never be committed.

`vision_model` is the preferred model for `image.analyze`; set it to `null` to let the runtime pick a vision-capable model from the catalog.

## Loop Parameters

| Key | Meaning |
|-----|---------|
| `loop.max_concurrent_ticks` | upper bound for ticks that may run at the same time; `1` keeps the runtime fully serial |
| `loop.max_tick_queue` | bounded dispatcher queue size for pending ticks; when full, chat requests are released back to pending and retried in a later cycle |
| `loop.max_idle_gap` | default idle wait ceiling in milliseconds when there is no active work |
| `loop.active_idle_gap` | default wait interval in milliseconds while a task is active |
| `loop.min_act_gap` | minimum interval in milliseconds between two `act` decisions |
| `loop.judge_every` | when fully idle, only call the LLM every N ticks; ignored when there is active work or a user message |
| `loop.max_consecutive_errors` | consecutive error threshold |
| `loop.evolve_every` | evolution check frequency |

### Concurrent Tick Constraints

- `max_concurrent_ticks` only unlocks concurrency for ticks that do not share continuation state.
- If a tick depends on the previous round's `next_step`, `last_action_*`, `pending_tier`, or stall counters, it must remain queued behind the same chain.
- A safe rollout starts with `max_concurrent_ticks=2`; keeping `1` preserves the old fully serial behavior.

## Memory Parameters

| Key | Meaning |
|-----|---------|
| `memory.working_capacity` | working memory item capacity |
| `memory.max_events` | episodic event cap |
| `memory.semantic_decay_lambda` | semantic memory decay coefficient |
| `memory.embedding_provider` | independent embedding provider; `local` uses local embeddings, `none` disables vector embedding, other values select a named provider |
| `memory.embedding_model` | embedding model ID, independent from the main chat model when `embedding_provider` is set |
| `memory.embedding_fallback` | fallback behavior when embeddings are unavailable; default keeps FTS5/text retrieval |
| `memory.embedding_weight` | hybrid retrieval vector weight |

## Evolution Parameters

| Key | Meaning |
|-----|---------|
| `evolution.enabled` | enable or disable self-evolution |
| `evolution.trigger_min_failures` | minimum failures inside a trigger window |
| `evolution.trigger_window_minutes` | failure window size |
| `evolution.error_streak_evolve` | immediate trigger on error streak |
| `evolution.max_attempts` | retries per evolution run |
| `evolution.backup` | whether to back up files before patching |

## Gateway Parameters

| Key | Meaning |
|-----|---------|
| `gateway.default_channel` | default ingress channel such as `local` or `wechat` |

## Environment Variables

Environment variables are still the preferred production path.

Typical `.env` entries:

```bash
DASHSCOPE_API_KEY=...
DEEPSEEK_API_KEY=...
COPILOT_GITHUB_TOKEN=...
```

## CLI Helpers

### Minimal starter config

```bash
cp lingzhou.min.json.example ~/.lingzhou/lingzhou.json
# Fill in the model name; everything else uses built-in defaults.
```

`lingzhou.min.json.example` contains only the required fields; all other values are derived from module defaults at runtime.

### Discover config keys

```bash
lingzhou config keys              # list all groups
lingzhou config keys loop         # keys and current values for the loop group
lingzhou config keys memory       # memory group
lingzhou config keys --defaults   # include default column
```

### IDE autocomplete via JSON Schema

```bash
lingzhou config schema -o lingzhou-schema.json
```

Then wire it in VS Code `settings.json`:

```jsonc
{
  "json.schemas": [
    {
      "fileMatch": ["lingzhou.json"],
      "url": "./lingzhou-schema.json"
    }
  ]
}
```

Your `lingzhou.json` file will then have inline validation and completion.

The example file [lingzhou.json.example](../lingzhou.json.example) contains a more complete configuration skeleton.
