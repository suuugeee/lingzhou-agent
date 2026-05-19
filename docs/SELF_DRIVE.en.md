# Self-Drive and Autonomous Exploration

[中文](SELF_DRIVE.md) | [English](SELF_DRIVE.en.md)

## Goal

Self-drive is not a cron job and not a hardcoded patrol loop. Its job is to decide, during idle periods, whether the system should keep exploring, reorganize memory, repair itself, or stay quiet.

## Three Core Signals

Autonomous behavior is mainly driven by three internal signals:

| Signal | Meaning | Typical Sources |
|--------|---------|-----------------|
| Novelty | whether the system is encountering new objects, paths, or knowledge | new files, new web pages, new tasks, new semantic nodes |
| Learning Progress | whether recent work produced real capability gains or closed loops | completed tasks, structural reflection, successful repairs |
| Surprise | whether reality diverges from prediction | tool failures, probe anomalies, missing config, behavioral loops |

These signals do not replace judgment. They feed into perception first, then shape later decisions.

## Position in the Cognition Loop

```text
Perception -> Self-Drive -> Judgment -> Execution -> Reflection
```

Self-drive answers two questions:

1. Is there a reason to act now?
2. If yes, should the system explore, verify, reorganize, or wait?

It does not directly execute tools. It only influences the next judgment decision.

## When Autonomous Behavior Tends to Trigger

Common trigger situations:

- no new user message, but unresolved high-value evidence remains
- repeated failures require reflection or evidence gathering
- a probe detects external state changes
- the system has been idle for a while, but there is still meaningful exploration or consolidation work left

Common non-trigger situations:

- the user is actively waiting for a reply
- repeated exploration has already stalled with no new evidence
- the missing condition is external and cannot be resolved locally

## Relationship to Memory

Self-drive depends heavily on memory quality:

- high working-memory pressure pushes toward consolidation instead of expansion
- low semantic retrieval quality pushes toward evidence gathering instead of direct conclusions
- repeated episodic failures increase surprise and trigger strategy change

So self-drive is not isolated. It is tightly coupled to the memory stack.

## Relevant Configuration

Important config keys include:

- `loop.max_idle_gap`
- `loop.active_idle_gap`
- `thresholds.curiosity_idle_task`
- `thresholds.curiosity_idle_min_cycles`
- `memory.semantic_decay_lambda`
- `memory.chat_crystallize_every`

These settings define how conservative or proactive lingzhou behaves during idle time.

## Current Boundaries

- autonomous exploration still obeys task boundaries and hard safety rules
- high-cost actions must still pass through judgment
- user-facing work takes priority when a reply is pending