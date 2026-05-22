# Configuration Reference

[中文](CONFIG.md) | [English](CONFIG.en.md)

By default, configuration lives in `~/.lingzhou/lingzhou.json`. The runtime can inspect and update config through `config.get` and `config.set`, and then reload it without restarting the whole process.

The default runtime directory layout is under `~/.lingzhou/`: `db_path`, `memory_dir`, `state_dir`, `workspace_dir`, logs, and temporary artifacts should stay there in production instead of being written back into the source repository.

## Model Routing

```jsonc
{
  "model": "bailian/qwen3.6-plus",
  "routing": {
    "reader": "bailian/qwen-plus",
    "reasoner": "copilot/gpt-5.4",
    "repair": "bailian/qwen3.6-plus"
  }
}
```

Providers are defined in the `providers` section. API keys should come from environment variables or auth profiles, not from committed config files.

## Loop Parameters

| Key | Meaning |
|-----|---------|
| `loop.max_concurrent_ticks` | upper bound for ticks that may run at the same time; `1` keeps the runtime fully serial |
| `loop.max_tick_queue` | bounded dispatcher queue size for pending ticks; chat waits for a slot instead of returning busy |
| `loop.max_idle_gap` | default idle wait ceiling in milliseconds when there is no active work |
| `loop.active_idle_gap` | default wait interval in milliseconds while a task is active |
| `loop.min_act_gap` | minimum interval in milliseconds between two `act` decisions |
| `loop.chat_reply_timeout` | timeout for chat reply waiting |
| `loop.max_tool_rounds` | max tool rounds inside a single tick |
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

Typical `.env` entries:

```bash
DASHSCOPE_API_KEY=...
DEEPSEEK_API_KEY=...
COPILOT_GITHUB_TOKEN=...
```

The example file [lingzhou.json.example](../lingzhou.json.example) contains a more complete configuration skeleton.