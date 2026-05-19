# 插件开发指南

[中文](PLUGIN.md) | [English](PLUGIN.en.md)

## 快速开始

```bash
# 创建插件骨架
lingzhou gateway plugin install my-plugin

# 目录结构
plugins/my-plugin/
├── plugin.json      # 元数据
└── __init__.py      # 入口
```

## plugin.json

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "0.1.0",
  "description": "插件描述",
  "channels": [],
  "tools": ["my.tool"]
}
```

## __init__.py

```python
def register(ctx):
    """注册工具/通道。ctx: PluginContext"""
    from tools.registry import tool, ToolManifest, ToolResult

    @tool(ToolManifest(
        name="my.tool",
        description="我的工具",
        params=[],
    ))
    async def my_tool(params, ctx):
        return ToolResult(summary="Hello from plugin!")

def unregister():
    """清理资源（可选）"""
    pass

def start():
    """插件启动时调用（可选）"""
    pass

def stop():
    """插件停止时调用（可选）"""
    pass
```

## PluginContext

传递给 `register(ctx)` 的上下文：

```python
@dataclass
class PluginContext:
    manifest: PluginManifest     # 插件元数据
    tool_registry: Any = None    # 工具注册表（可注册新工具）
    channel_registry: Any = None # 通道注册表（可注册新通道）
```

## 生命周期

```
discover → load → register → start → [运行] → stop → unregister
```

灵舟启动时自动执行 discover→load→register→start，
关闭时执行 stop→unregister。

## 管理命令

```bash
lingzhou gateway plugin list          # 列出已安装
lingzhou gateway plugin install xxx   # 安装
lingzhou gateway plugin install xxx -s /path  # 从源码
lingzhou gateway plugin remove xxx    # 移除
```
