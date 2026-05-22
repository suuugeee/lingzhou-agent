# Architecture

[中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md)

## Cognition Loop

```text
         ┌──────────────┐
         │ Perception   │ ← working memory + episodic memory + prediction error
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Self-Drive   │ ← novelty + learning progress + surprise
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Judgment     │ ← LLM decisions (act / wait / pause) + tool choice
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Execution    │ ← built-in tools + inner continue loop
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Reflection   │ ← episodic consolidation + semantic compilation + emotion update
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Evolution    │ ← failure patterns → generated fixes → hot reload
         └──────────────┘
```

## Tick Dispatch and Ordering Boundaries

The goal of concurrent ticks is not "run everything out of order". It is "run unrelated ticks concurrently while preserving order for related work".

- Ticks that belong to the same task continuation chain must stay FIFO. Their ordering depends on cross-tick state such as `next_step`, `last_action_*`, `pending_tier`, `ticks_since_judge`, and stall counters.
- Ticks from unrelated tasks, or ticks that do not share continuation state, may run concurrently to reduce chat starvation while an autonomous LLM call is still in flight.
- The runtime should enforce this through a bounded dispatcher: serialize per chain, run chains concurrently under a global concurrency cap, and keep a bounded pending queue.
- The design goal is better responsiveness and throughput without changing the causal order of related work.

## Core Modules

### `core/loop/runtime.py` — Main loop

Coordinates the full perception → judgment → execution → reflection cycle. The runtime is event-driven and wakes on chat messages, task changes, or timeout boundaries.

With concurrent ticks enabled, the runtime is also responsible for:

- owning global shared resources such as the provider, task store, and memory layers
- dispatching ticks into continuation chains
- preserving FIFO inside one chain while allowing bounded cross-chain concurrency

### `core/judgment/runtime.py` — Judgment layer

This is the LLM decision engine. It receives working memory, runtime signals, task context, and tool manifests, then decides what to do next.

Key features:

- multi-model routing for `reader`, `reasoner`, and `repair`
- continuation loops that reuse cached context instead of rebuilding it every tool round
- manifest-driven tool routing and capability governance

### `core/perception/` — Perception layer

Builds perceptual state from memory, emotion, and recent execution feedback.

Submodules:

- `emotion.py`: OCC-style emotion state and replay summary
- `ethos.py`: value state and ethical baseline
- `signals.py`: judgment and cognitive signals
- `layer.py`: perception entry point

### `core/self_drive.py` — Self-drive engine

Evaluates whether the system should stay quiet, continue exploring, or reorganize itself during idle periods.

### `core/evolution.py` — Evolution engine

Detects failure patterns, asks the model to synthesize a patch, validates the result, hot-reloads the change, and rolls back when validation fails.

### `core/behavior_tracker.py` — Behavior tracker

Tracks repetitive actions and exploration loops, then writes those signals back into working memory for the model to notice.

### `core/plugin.py` — Plugin manager

Implements the lifecycle `discover → load → register → start`, and unload flow on shutdown.

## Memory System

### Working memory

Short-horizon memory that is eligible for direct prompt injection.

### Episodic memory

Append-only event history for turns, actions, and outcomes.

### Semantic memory

Long-term reusable knowledge compiled into searchable nodes.

### Task store

Persistent SQLite-backed storage for tasks, chat messages, failures, facts, signals, runs, and meta reflections.

## Tool System

All Python files under `tools/` are auto-discovered. Every tool is declared with `@tool(ToolManifest(...))`, returns `ToolResult`, and is registered into the shared tool registry.

## Channel Architecture

lingzhou currently supports three IO modes:

- `local` — terminal chat
- `wechat` — WeChat iLink channel
- `webhook` — HTTP integration

These channels run alongside the cognition loop and inject events into the runtime.