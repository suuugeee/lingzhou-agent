"""cli/dev.py — dev 子命令组：evolve / tools / skills / model / update / version / doctor。"""
from __future__ import annotations

import asyncio
import json as _json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from cli import dev_helpers as _dev_helpers
from cli.common import DEFAULT_CONFIG_PATH, PROJECT_ROOT, console, load_cfg
from core.judgment.tiers import JUDGMENT_TIERS, tier_display_label
from cli.dev_helpers import (
    _apply_model_target_selection,
    _effective_target_model,
    _model_supports_vision,
    _normalize_model_target,
    _preferred_model_index,
    _set_db_routing_override,
    _sync_db_routing_overrides,
)
from cli.diag import doctor, version
from core.version import __codename__, __version__

_merge_runtime_routing_override = _dev_helpers._merge_runtime_routing_override
_sync_routing_models_on_primary_switch = _dev_helpers._sync_routing_models_on_primary_switch

dev_app = typer.Typer(
    name="dev",
    help="开发者工具：evolve / tools / skills / model / update / version / doctor",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dev_app.command("evolve")
def evolve(
    description: Annotated[str, typer.Argument(help="新工具的自然语言描述")],
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """合成并热加载一个新工具（自进化）。"""
    cfg = load_cfg(config)

    async def _run() -> None:
        from core.evolution import EvolutionEngine
        from provider import create_provider
        from tools.registry import ToolRegistry

        provider = create_provider(cfg)
        registry = ToolRegistry()
        engine = EvolutionEngine(cfg, provider, registry)
        result = await engine.synthesize_tool(description)
        await provider.close()
        if result.success:
            console.print(f"[green]工具 {result.target!r} 已合成[/green]")
        else:
            console.print(f"[red]合成失败: {result.reason}[/red]")

    asyncio.run(_run())


@dev_app.command("tools")
def tools(
    search: Annotated[str | None, typer.Argument(help="关键词过滤")] = None,
) -> None:
    """列出所有已注册的工具（支持关键词过滤）。"""
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    tools_dir = PROJECT_ROOT / "tools"
    reg.discover(tools_dir)
    manifests = reg.list_manifests()

    if search:
        kw = search.lower()
        manifests = [
            m for m in manifests
            if kw in m.name.lower() or kw in (m.description or "").lower()
        ]

    if not manifests:
        console.print("（没有匹配的工具）")
        return

    console.print(f"[bold]已注册工具[/bold]  ({len(manifests)} 个)\n")
    for m in sorted(manifests, key=lambda x: x.name):
        console.print(f"  [cyan]{m.name:<26}[/cyan] {m.description or ''}")


@dev_app.command("compact-runtime")
def compact_runtime(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    db_path: Annotated[Path | None, typer.Option("--db", help="runtime.db 路径；默认读取配置中的 loop.db_path")] = None,
    apply_changes: Annotated[bool, typer.Option("--apply/--dry-run", help="实际写入压缩结果；默认 dry-run")] = False,
    vacuum: Annotated[bool, typer.Option("--vacuum/--no-vacuum", help="apply 后执行 VACUUM 回收文件空间")] = False,
) -> None:
    """压缩 runtime.db 中 oversized run/task/fact/ledger 载荷。"""
    from core.maintenance import compact_runtime_db

    cfg = load_cfg(config)
    target = db_path or Path(str(cfg.loop.db_path)).expanduser()
    report = compact_runtime_db(target, apply=apply_changes, vacuum=vacuum)
    if report.get("error"):
        console.print(f"[red]runtime compact failed:[/red] {report['error']}  {report['db_path']}")
        raise typer.Exit(1)

    mode = "apply" if apply_changes else "dry-run"
    console.print(f"[bold]runtime compact[/bold] mode={mode} db={report['db_path']}")
    console.print(
        "  scanned={scanned_rows} changed={changed_rows} saved≈{saved_bytes} bytes".format(**report)
    )
    if apply_changes:
        console.print(
            "  file_bytes_before={file_bytes_before} file_bytes_after={file_bytes_after} vacuumed={vacuumed}".format(**report)
        )
    for table, stats in sorted((report.get("tables") or {}).items()):
        if not stats.get("changed_rows"):
            continue
        console.print(
            "  {table}: changed={changed_rows} {original_bytes}->{compacted_bytes} bytes".format(
                table=table,
                **stats,
            )
        )
    if not apply_changes:
        console.print("[yellow]未写入数据库。确认结果后加 --apply；需要回收文件空间再加 --vacuum。[/yellow]")


@dev_app.command("compact-memory")
def compact_memory(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    memory_dir: Annotated[Path | None, typer.Option("--memory-dir", help="语义记忆目录；默认读取配置中的 loop.memory_dir")] = None,
    apply_changes: Annotated[bool, typer.Option("--apply/--dry-run", help="实际写入压缩结果；默认 dry-run")] = False,
    vacuum: Annotated[bool, typer.Option("--vacuum/--no-vacuum", help="apply 后对 memory SQLite DB 执行 VACUUM")] = False,
) -> None:
    """压缩 memory JSON、episodic markdown 与 semantic/episodic SQLite DB 中 oversized 文本。"""
    from core.maintenance import compact_memory_dir

    cfg = load_cfg(config)
    target = memory_dir or cfg.memory_dir
    report = compact_memory_dir(target, apply=apply_changes, vacuum=vacuum)
    if report.get("error"):
        console.print(f"[red]memory compact failed:[/red] {report['error']}  {report['memory_dir']}")
        raise typer.Exit(1)

    mode = "apply" if apply_changes else "dry-run"
    console.print(f"[bold]memory compact[/bold] mode={mode} dir={report['memory_dir']}")
    console.print(
        (
            "  files scanned={scanned_files} changed={changed_files} bad_json={bad_json_files}; "
            "db_rows scanned={scanned_rows} changed={changed_rows}; saved≈{saved_bytes} bytes"
        ).format(**report)
    )
    if apply_changes:
        console.print("  vacuumed={vacuumed}".format(**report))
    for dirname, stats in sorted((report.get("dirs") or {}).items()):
        if not stats.get("changed_files") and not stats.get("bad_json_files"):
            continue
        console.print(
            "  {dirname}: changed={changed_files} bad_json={bad_json_files} {original_bytes}->{compacted_bytes} bytes".format(
                dirname=dirname,
                **stats,
            )
        )
    for db_name, stats in sorted((report.get("dbs") or {}).items()):
        if not stats.get("changed_rows"):
            continue
        console.print(
            "  {db}: changed_rows={changed_rows} {original_bytes}->{compacted_bytes} bytes vacuumed={vacuumed}".format(
                db=db_name,
                **stats,
            )
        )
    if not apply_changes:
        console.print("[yellow]未写入文件/数据库。确认结果后加 --apply；需要回收 DB 文件空间再加 --vacuum。[/yellow]")


@dev_app.command("skills")
def skills(
    search: Annotated[str | None, typer.Argument(help="关键词过滤")] = None,
    disabled: Annotated[bool, typer.Option("--disabled", help="显示已禁用 skills，而不是 active skills")] = False,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """列出当前 workspace 中可被运行态加载的 skills。"""
    from core.skill import SkillRegistry

    cfg = load_cfg(config)
    skills_dir = cfg.workspace_dir / ("skills-disabled" if disabled else "skills")
    reg = SkillRegistry(skills_dir=skills_dir)
    items = [s for s in reg.all_skills() if getattr(s, "origin", "builtin") == "workspace"]

    if search:
        kw = search.lower()
        items = [
            s for s in items
            if kw in s.name.lower()
            or kw in (s.description or "").lower()
            or kw in " ".join(s.triggers).lower()
        ]

    state = "disabled" if disabled else "active"
    if not items:
        console.print(f"（没有匹配的 {state} skills）")
        return

    console.print(f"[bold]{state} skills[/bold]  ({len(items)} 个)  [dim]{skills_dir}[/dim]\n")
    for s in sorted(items, key=lambda x: x.name):
        trig = f"  [dim]triggers: {', '.join(s.triggers[:6])}[/dim]" if s.triggers else ""
        console.print(f"  [cyan]{s.name:<24}[/cyan] {s.description}{trig}")


@dev_app.command("model")
def model(
    set_model: Annotated[str | None, typer.Argument(help="要切换的模型 ID，如 bailian/qwen-plus")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="列出所有可用模型")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="交互式选择 provider 和模型")] = False,
) -> None:
    """查看或切换当前使用的 LLM provider / 模型。"""
    from provider.catalog import list_provider_models, list_providers

    if list_all:
        for pname in list_providers():
            models_list = list_provider_models(pname)
            console.print(f"\n[bold]{pname}[/bold]")
            for m in models_list:
                ctx_k = (m.get("context_window") or 0) // 1000
                tags = []
                if m.get("thinking"):
                    tags.append("thinking")
                if m.get("reasoning"):
                    tags.append("reasoning")
                if _model_supports_vision(m):
                    tags.append("vision")
                tag_str = f"  [dim][{', '.join(tags)}][/dim]" if tags else ""
                ctx_str = f"  [dim]{ctx_k}K[/dim]" if ctx_k else ""
                console.print(f"  {m['id']}{ctx_str}{tag_str}")
        return

    cfg_path = config if config.exists() else None
    # 尝试在搜索路径中找到配置
    if cfg_path is None:
        from cli.common import find_config
        try:
            cfg_path = find_config(config)
        except SystemExit:
            cfg_path = None

    if cfg_path is None or not cfg_path.exists():
        console.print(f"[red]配置文件不存在: {config}，请先运行 lingzhou setup[/red]")
        raise typer.Exit(1)

    cfg_data = _json.loads(cfg_path.read_text(encoding="utf-8"))
    current = cfg_data.get("model", "(未设置)")
    chosen_thinking = cfg_data.get("thinking", "off")
    model_target = "primary"
    current_target_model = str(current)
    current_provider, _, current_model_id = current_target_model.partition("/")

    # ── 交互式选择 ─────────────────────────────────────────────────────────
    if interactive or (not set_model):
        console.print(f"当前模型: [bold cyan]{current}[/bold cyan]")
        if not interactive:
            console.print("[dim]切换模型: lingzhou model <provider/model-id>[/dim]")
            console.print("[dim]交互切换: lingzhou dev model -i[/dim]")
            console.print("[dim]查看全部: lingzhou model --list[/dim]")
            return

        console.print("\n[bold]选择要设置的模型槽位[/bold]")
        target_options = [
            ("primary", f"主模型 (model)  [dim]当前: {current}[/dim]"),
            ("vision", f"识图模型 (vision_model)  [dim]当前: {_effective_target_model(cfg_data, 'vision')}[/dim]"),
        ] + [
            (tier, f"{tier_display_label(tier)} ({tier})  [dim]当前: {_effective_target_model(cfg_data, tier)}[/dim>")
            for tier in JUDGMENT_TIERS
        ] + [("other", "其他 routing 键（手动输入）")]
        for i, (_, label) in enumerate(target_options, 1):
            console.print(f"  {i}. {label}")

        raw_target = typer.prompt("模型槽位编号", default="1")
        try:
            target_idx = int(raw_target.strip()) - 1
        except ValueError:
            target_idx = 0
        if not (0 <= target_idx < len(target_options)):
            console.print("[red]无效编号[/red]")
            raise typer.Exit(1)

        selected_target = target_options[target_idx][0]
        if selected_target == "other":
            entered_key = typer.prompt("  输入 routing 键", default=JUDGMENT_TIERS[1])
            model_target = _normalize_model_target(entered_key)
        else:
            model_target = selected_target

        current_target_model = _effective_target_model(cfg_data, model_target)
        current_provider, _, current_model_id = current_target_model.partition("/")
        console.print(
            f"[dim]本次将设置 {_normalize_model_target(model_target)} 模型槽位，当前值: {current_target_model or current}[/dim]"
        )

        # 交互式：先选 provider
        configured_providers = list(cfg_data.get("providers", {}).keys())
        all_catalog = list_providers()
        # 配置了的 provider 排在前面
        ordered = configured_providers + [p for p in all_catalog if p not in configured_providers]

        console.print("\n[bold]选择 provider[/bold]")
        for i, p in enumerate(ordered, 1):
            mark = "[green]✓[/green]" if p in configured_providers else "[dim]  [/dim]"
            console.print(f"  {i}. {mark} {p}")

        raw_p = typer.prompt("Provider 编号", default="1")
        try:
            pidx = int(raw_p.strip()) - 1
        except ValueError:
            pidx = 0
        if not (0 <= pidx < len(ordered)):
            console.print("[red]无效编号[/red]")
            raise typer.Exit(1)
        chosen_provider = ordered[pidx]

        # 如果选了未配置的 provider，引导用户补充配置并写入 lingzhou.json
        if chosen_provider not in configured_providers:
            _BUILTIN_PROVIDER_DEFAULTS: dict[str, dict] = {
                "bailian": {
                    "type": "openai_compat",
                    "mode": "openai",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key_env": "DASHSCOPE_API_KEY",
                },
                "deepseek": {
                    "type": "openai_compat",
                    "mode": "openai",
                    "base_url": "https://api.deepseek.com",
                    "api_key_env": "DEEPSEEK_API_KEY",
                },
                "copilot": {
                    "type": "openai_compat",
                    "mode": "copilot",
                    "base_url": "https://api.individual.githubcopilot.com",
                    "api_key_env": "GITHUB_TOKEN",
                },
            }
            defaults = _BUILTIN_PROVIDER_DEFAULTS.get(chosen_provider, {
                "type": "openai_compat",
                "mode": "openai",
                "base_url": "",
                "api_key_env": "OPENAI_API_KEY",
            })

            new_provider_cfg: dict = {
                "type": defaults["type"],
                "mode": defaults["mode"],
                "base_url": defaults["base_url"],
                "api_key_env": defaults["api_key_env"],
                "auth_profile_id": f"{chosen_provider}:default",
            }

            # copilot 走 auth login 的 token exchange 链，不需要手动填 key
            if chosen_provider == "copilot":
                from store.auth import COPILOT_PROFILE_ID, get_auth_profile
                existing_auth = get_auth_profile(COPILOT_PROFILE_ID)
                if existing_auth and existing_auth.get("token"):
                    console.print("\n[green]✓ 已检测到 Copilot 登录凭证[/green]  [dim](lingzhou auth login 已完成)[/dim]")
                else:
                    console.print("\n[yellow]Copilot 尚未登录[/yellow]")
                    console.print("  请在切换后运行: [bold]lingzhou auth login[/bold]")
            else:
                # 其他 provider 需要手动输入 API key 或环境变量名
                import re as _re
                console.print(f"\n[yellow]{chosen_provider} 未在配置中，现在为你补充配置。[/yellow]")
                api_key_input = typer.prompt(
                    "  环境变量名或直接粘贴 API key",
                    default=defaults["api_key_env"],
                )
                new_provider_cfg["api_key_env"] = api_key_input
                # 如果输入的不是 ENV_VAR 格式（直接贴了 key），写入 auth profile。
                if api_key_input and not _re.match(r'^[A-Z_][A-Z0-9_]*$', api_key_input.strip()):
                    from store.auth import AUTH_PROFILES_PATH, set_token_profile

                    cred_key = f"{chosen_provider.upper()}_API_KEY"
                    profile_id = f"{chosen_provider}:default"
                    set_token_profile(profile_id=profile_id, provider=chosen_provider, token=api_key_input.strip())
                    new_provider_cfg["api_key_env"] = cred_key
                    new_provider_cfg["auth_profile_id"] = profile_id
                    console.print(f"  [dim]key 已安全存入 {AUTH_PROFILES_PATH}，配置中使用 {profile_id}[/dim]")

            if "providers" not in cfg_data:
                cfg_data["providers"] = {}
            cfg_data["providers"][chosen_provider] = new_provider_cfg
            configured_providers.append(chosen_provider)
            console.print(f"[green]✓ {chosen_provider} 已添加到配置[/green]")

        # 选模型
        catalog_models = list_provider_models(chosen_provider)
        if _normalize_model_target(model_target) == "vision":
            vision_models = [m for m in catalog_models if _model_supports_vision(m)]
            if vision_models:
                catalog_models = vision_models
        console.print(f"\n[bold]选择模型[/bold]  [dim](provider={chosen_provider})[/dim]")
        if catalog_models:
            preferred_index = _preferred_model_index(
                catalog_models,
                current_model_id=current_model_id if chosen_provider == current_provider else "",
            )
            for i, m in enumerate(catalog_models, 1):
                ctx_k = (m.get("context_window") or 0) // 1000
                tags = []
                if m.get("thinking"):
                    tags.append("thinking")
                if m.get("reasoning"):
                    tags.append("reasoning")
                if _model_supports_vision(m):
                    tags.append("vision")
                ctx_str = f"  [dim]{ctx_k}K[/dim]" if ctx_k else ""
                tag_str = f"  [dim][{', '.join(tags)}][/dim]" if tags else ""
                mark = " [bold cyan]← 默认[/bold cyan]" if i - 1 == preferred_index else ""
                console.print(f"  {i}. {m['id']}{ctx_str}{tag_str}{mark}")
            console.print(f"  {len(catalog_models)+1}. 手动输入")
            raw_m = typer.prompt("  模型编号", default=str(preferred_index + 1))
            try:
                midx = int(raw_m.strip()) - 1
            except ValueError:
                midx = -1
            if 0 <= midx < len(catalog_models):
                chosen_model_id = catalog_models[midx]["id"]
            else:
                chosen_model_id = typer.prompt("  手动输入模型 ID")
        else:
            chosen_model_id = typer.prompt(f"  {chosen_provider} 模型 ID")

        set_model = f"{chosen_provider}/{chosen_model_id}"

        if _normalize_model_target(model_target) == "primary":
            # 选思考等级
            _THINKING_LEVELS = ["off", "minimal", "low", "medium", "high"]
            _THINKING_DESC = {
                "off":     "关闭思考，速度最快，省 token",
                "minimal": "极浅思考，轻量推理",
                "low":     "低强度思考，例行决策",
                "medium":  "中等思考，常规判断（推荐日常）",
                "high":    "深度思考，复杂推理/代码生成",
            }
            current_thinking = cfg_data.get("thinking", "off")
            console.print(f"\n[bold]选择思考等级[/bold]  [dim](当前: {current_thinking})[/dim]")
            for i, lvl in enumerate(_THINKING_LEVELS, 1):
                mark = "[bold cyan]●[/bold cyan]" if lvl == current_thinking else " "
                console.print(f"  {i}. {mark} {lvl:<8} [dim]{_THINKING_DESC[lvl]}[/dim]")
            cur_default = str(_THINKING_LEVELS.index(current_thinking) + 1) if current_thinking in _THINKING_LEVELS else "1"
            raw_t = typer.prompt("  等级编号", default=cur_default)
            try:
                tidx = int(raw_t.strip()) - 1
            except ValueError:
                tidx = -1
            chosen_thinking = _THINKING_LEVELS[tidx] if 0 <= tidx < len(_THINKING_LEVELS) else current_thinking

    # ── 写入配置 ───────────────────────────────────────────────────────────
    selection = _apply_model_target_selection(
        cfg_data,
        current_model=str(current),
        new_model=str(set_model),
        target=model_target,
    )
    synced_routing = selection["routing_changed"]
    if not interactive or selection["target"] != "primary":
        chosen_thinking = cfg_data.get("thinking", "off")  # 非交互模式保持原值

    old_thinking = cfg_data.get("thinking", "off")
    cfg_data["thinking"] = chosen_thinking
    cfg_path.write_text(_json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")
    if selection["target"] == "primary":
        # 同步 DB routing_overrides：避免重启后从 DB 恢复到旧模型
        _sync_db_routing_overrides(cfg_path, old_model=str(current), new_model=str(set_model))
        console.print(f"[green]✓ 模型已切换:[/green] {current} → [bold cyan]{set_model}[/bold cyan]")
    else:
        console.print(
            f"[green]✓ {selection['target']} 模型已更新:[/green]"
            f" {selection['previous'] or current} → [bold cyan]{set_model}[/bold cyan]"
        )
        runtime_tier = selection["runtime_override_tier"]
        if runtime_tier:
            _set_db_routing_override(cfg_path, tier=runtime_tier, model_ref=str(set_model))
        else:
            console.print(
                f"[yellow]⚠ routing 键 {selection['target']} 不是标准运行时 tier；仅已写入 config routing。[/yellow]"
            )
    if synced_routing and selection["target"] == "primary":
        console.print(
            f"[green]✓ 已同步 routing:[/green] {', '.join(synced_routing)} → [bold cyan]{set_model}[/bold cyan]"
        )
    if chosen_thinking != old_thinking:
        console.print(f"[green]✓ 思考等级已更新:[/green] {old_thinking} → [bold cyan]{chosen_thinking}[/bold cyan]")
    console.print("[dim]lingzhou 运行中时将在下一轮自动生效（配置热重载）[/dim]")


@dev_app.command("update")
def update() -> None:
    """更新 lingzhou 到最新版本（git pull + 重新安装依赖）。"""
    console.print(f"当前版本: [bold]v{__version__}[/bold]  代号: {__codename__}")

    repo_dir = PROJECT_ROOT
    if not (repo_dir / ".git").exists():
        console.print("[yellow]当前目录不是 git 工作区，请手动拉取最新代码后重新安装：[/yellow]")
        console.print("  git pull && uv pip install -e .")
        raise typer.Exit(1)

    console.print("[dim]执行 git pull...[/dim]")
    result = subprocess.run(["git", "pull"], cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]git pull 失败:[/red]\n{result.stderr.strip()}")
        raise typer.Exit(1)
    console.print(f"[green]{result.stdout.strip() or 'Already up to date.'}[/green]")

    uv = shutil.which("uv")
    pip_cmd = [uv, "pip", "install", "-e", "."] if uv else [
        shutil.which("pip") or "pip", "install", "-e", "."
    ]
    console.print(f"[dim]重装依赖: {' '.join(pip_cmd)}[/dim]")
    result = subprocess.run(pip_cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]✓ 更新完成，重启 lingzhou 生效[/green]")
    else:
        console.print(f"[red]依赖安装失败:[/red]\n{result.stderr.strip()}")
        raise typer.Exit(1)


dev_app.command("version")(version)
dev_app.command("doctor")(doctor)
