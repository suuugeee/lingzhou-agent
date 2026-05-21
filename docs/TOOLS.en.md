# Tool Catalog

[中文](TOOLS.md) | [English](TOOLS.en.md)

## Overview

The repository exposes built-in tool endpoints that are registered with `@tool(ToolManifest(...))` and auto-discovered by the runtime. The total count evolves over time, so the live discover result is the source of truth.

Two design rules matter:

1. tools execute actions, they do not speak to the user on behalf of the model
2. tool behavior is governed by manifest metadata and capabilities rather than fragile tool-name hardcoding

## Categories

### 1. Files and Config

- `file.list`
- `file.read`
- `file.write`
- `file.edit`
- `file.delete`
- `config.get`
- `config.set`

### 2. Shell and Processes

- `shell.run`
- `shell.capabilities`
- `exec`
- `process.list`
- `process.poll`
- `process.log`
- `process.write`
- `process.kill`

### 3. Tasks, Planning, and Scheduling

- `task.add`
- `task.advance`
- `task.complete`
- `task.list`
- `task.update`
- `task.fail`
- `task.wait`
- `task.resume`
- `task.steer`
- `task.ask`
- `task.plan`
- `schedule.add`
- `schedule.list`
- `schedule.ack`
- `schedule.cancel`

### 4. Memory and Reflection

- `memory.add_wm`
- `memory.drop_wm`
- `memory.add_semantic`
- `memory.set_fact`
- `memory.search`
- `memory.get_fact`
- `memory.snapshot`
- `failure.dismiss`
- `reflect.structural`

### 5. Web, Browser, and Media

- `web.fetch`
- `web.search`
- `browser.navigate`
- `browser.snapshot`
- `browser.click`
- `browser.type`
- `browser.scroll`
- `image.analyze`
- `image.generate`
- `tts.speak`

### 6. Skills, Probes, and Notifications

- `skill.list`
- `skill.search`
- `probe.install`
- `probe.remove`
- `probe.run`
- `probe.list`
- `probe.disable`
- `probe.enable`
- `wechat.send`

## Capability Tags

Some tools declare capabilities used by the judgment and execution layers:

| capability | Meaning |
|------------|---------|
| `ask_evidence` | local evidence-gathering tool that should be tried before asking the user |
| `plan_bootstrap_exempt` | allowed before a structured plan exists |
| `plan_alignment_exempt` | not blocked by plan-alignment checks |
| `completion_info_only` | read-only tool used in completion checks |
| `completion_mutation` | mutating tool used before completion |
| `completion_verify` | verification tool used before marking completion |

These tags let the runtime reason over tool intent instead of relying on hardcoded tool-name sets.

## ToolResult Contract

Tools return `ToolResult`, typically containing:

- `summary`
- `evidence`
- `error`
- `skipped`
- `state_delta`
- `metadata`

## When to Add a New Tool

Add a tool only when:

- the action has stable input/output boundaries
- the capability should exist as executable behavior rather than prompt prose
- the current tool surface cannot express the action cleanly

If the change is really about prompt policy, routing, or memory usage, change judgment logic or capabilities first.