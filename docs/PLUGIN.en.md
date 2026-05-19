# Plugin Development Guide

[中文](PLUGIN.md) | [English](PLUGIN.en.md)

## Quick Start

```bash
lingzhou gateway plugin install my-plugin

plugins/my-plugin/
├── plugin.json
└── __init__.py
```

## `plugin.json`

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "0.1.0",
  "description": "Plugin description",
  "channels": [],
  "tools": ["my.tool"]
}
```

## `__init__.py`

```python
def register(ctx):
    from tools.registry import tool, ToolManifest, ToolResult

    @tool(ToolManifest(
        name="my.tool",
        description="My tool",
        params=[],
    ))
    async def my_tool(params, ctx):
        return ToolResult(summary="Hello from plugin!")

def unregister():
    pass

def start():
    pass

def stop():
    pass
```

## PluginContext

`register(ctx)` receives a context object similar to:

```python
@dataclass
class PluginContext:
    manifest: PluginManifest
    tool_registry: Any = None
    channel_registry: Any = None
```

## Lifecycle

```text
discover → load → register → start → run → stop → unregister
```

On startup, lingzhou automatically performs discovery, load, register, and start. On shutdown it executes stop and unregister.

## Management Commands

```bash
lingzhou gateway plugin list
lingzhou gateway plugin install xxx
lingzhou gateway plugin install xxx -s /path
lingzhou gateway plugin remove xxx
```