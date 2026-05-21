"""core/skill.py — 技能系统（认知护栏 / Agent Skills 兼容层）。

技能不是工具：工具是执行能力，技能是注入 LLM 判断前的认知框架。
当前实现优先遵循 Agent Skills / OpenClaw 常见范式：
- discovery 只加载 metadata（name / description / location）
- activation 时才读取完整 SKILL.md
- scripts / references / assets 等资源按需再读

设计原则：
- 技能本身可以被 evolution 进化（本文件理论上可热替换）
- 标准载体是 workspace/skills/<name>/SKILL.md；保留对 legacy 单文件 *.md 的兼容读取
- judgment 默认只看到 catalog / 候选 skill 摘要，完整说明需显式 activation
- 自定义技能匹配仅基于 metadata，不基于整段 instruction 做硬编码触发
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("core.skill")

_SEED_SYNC_MANIFEST = ".seed-sync.json"


@dataclass
class SkillStateCondition:
    signal_name: str
    operator: str
    threshold: float


@dataclass
class SkillStateRule:
    signal_name: str = ""
    weight: float = 0.0
    conditions: list[SkillStateCondition] = field(default_factory=list)
    inhibit: bool = False


@dataclass
class SkillMatchRule:
    mode: str = "contains"
    terms: list[str] = field(default_factory=list)
    weight: float = 1.0
    inhibit: bool = False


# ---------- alias / migration layer -----------------------------------------
# 历史 dotted 名 → 规范 hyphen 名的映射表。
# workspace 里如果还存在旧目录，尽量通过这层透明寻找。
_SKILL_NAME_ALIASES: dict[str, str] = {
    "runtime.bootstrap":    "runtime-bootstrap",
    "failure.reflection":   "failure-reflection",
    "task.continuity":      "task-continuity",
    "provider.integration": "provider-integration",
}


def _canonical_skill_name(name: str) -> str:
    """dotted 历史名 → hyphen 规范名；已是规范名直接返回。"""
    return _SKILL_NAME_ALIASES.get(name, name)


def _seed_skills_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "skills"


def workspace_skill_file(workspace_dir: Path, skill_name: str) -> Path:
    return workspace_dir / "skills" / skill_name / "SKILL.md"


def ensure_workspace_skill_file(workspace_dir: Path, skill_name: str) -> Path:
    canonical = _canonical_skill_name(skill_name)
    target = workspace_skill_file(workspace_dir, canonical)
    if target.exists():
        return target

    # 兼容：如果 workspace 里还保留着旧 dotted 目录剪影
    if canonical != skill_name:
        dotted_target = workspace_skill_file(workspace_dir, skill_name)
        if dotted_target.exists():
            return dotted_target

    # 兼容： legacy 单文件（尝试 canonical 和 dotted 两种）
    for check_name in dict.fromkeys([canonical, skill_name]):
        legacy = workspace_dir / "skills" / f"{check_name}.md"
        if legacy.exists():
            return legacy

    seed_dir = _seed_skills_dir()
    candidates = [
        seed_dir / canonical / "SKILL.md",
        seed_dir / skill_name / "SKILL.md",
        seed_dir / f"{canonical}.md",
        seed_dir / f"{skill_name}.md",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _iter_skill_files(skills_dir: Path) -> list[Path]:
    files: list[Path] = []
    for md in sorted(skills_dir.glob("*.md")):
        if md.name != "SKILL.md":
            files.append(md)
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        files.append(skill_md)
    return files


def _skill_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _seed_sync_manifest_path(skills_dir: Path) -> Path:
    return skills_dir / _SEED_SYNC_MANIFEST


def _load_seed_sync_manifest(skills_dir: Path) -> dict[str, str]:
    manifest_path = _seed_sync_manifest_path(skills_dir)
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip()
    }


def _write_seed_sync_manifest(skills_dir: Path, manifest: dict[str, str]) -> None:
    manifest_path = _seed_sync_manifest_path(skills_dir)
    if not manifest:
        manifest_path.unlink(missing_ok=True)
        return
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sync_seed_skill_file(src: Path, dest: Path, *, relative: str, manifest: dict[str, str]) -> int:
    source_text = src.read_text(encoding="utf-8")
    source_hash = _skill_digest(source_text)

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source_text, encoding="utf-8")
        manifest[relative] = source_hash
        return 1

    dest_text = dest.read_text(encoding="utf-8")
    dest_hash = _skill_digest(dest_text)
    tracked_hash = manifest.get(relative, "")

    if dest_hash == source_hash:
        manifest[relative] = source_hash
        return 0

    if tracked_hash and dest_hash == tracked_hash:
        dest.write_text(source_text, encoding="utf-8")
        manifest[relative] = source_hash
        return 1

    manifest.pop(relative, None)
    return 0


def seed_workspace_skills(workspace_dir: Path) -> int:
    seed_dir = _seed_skills_dir()
    skills_dir = workspace_dir / "skills"
    if not seed_dir.exists():
        return 0
    written = 0
    skills_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_seed_sync_manifest(skills_dir)
    known_relatives: set[str] = set()
    for src in _iter_skill_files(seed_dir):
        relative_path = src.relative_to(seed_dir)
        relative = relative_path.as_posix()
        known_relatives.add(relative)
        dest = skills_dir / relative_path
        written += _sync_seed_skill_file(src, dest, relative=relative, manifest=manifest)
    for relative in list(manifest):
        if relative not in known_relatives:
            manifest.pop(relative, None)
    _write_seed_sync_manifest(skills_dir, manifest)
    if written:
        _log.info("[skill] 已向 %s 同步 %d 个默认 skills", skills_dir, written)
    return written


@dataclass
class Skill:
    name: str
    description: str      # 对人类的一句话说明（日志 / debug 用）
    guidance: str = ""   # 激活后才会读取 / 注入的完整 guidance
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    match_terms: list[str] = field(default_factory=list)
    match_rules: list[SkillMatchRule] = field(default_factory=list)
    state_bias: dict[str, float] = field(default_factory=dict)
    state_rules: list[SkillStateRule] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # 历史各字
    compatibility: str = ""
    license: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    origin: str = "dynamic"
    source_path: str = ""

    @property
    def is_standard_layout(self) -> bool:
        return Path(self.source_path).name == "SKILL.md"

    @property
    def skill_dir(self) -> Path:
        src = Path(self.source_path)
        return src.parent if src.name == "SKILL.md" else src.parent

    def load_markdown(self) -> str:
        if not self.source_path:
            return ""
        return Path(self.source_path).read_text(encoding="utf-8").strip()

    def load_guidance(self, limit: int | None = None) -> str:
        content = self.load_markdown()
        if not content:
            return ""
        _, body = _split_frontmatter(content)
        text = (body or content).strip()
        if limit is None or limit <= 0:
            return text
        return _trim_guidance(text, limit=limit)

    def list_resources(self, max_files: int = 20) -> list[str]:
        if not self.is_standard_layout:
            return []
        root = self.skill_dir
        files: list[str] = []
        for child in sorted(root.rglob("*")):
            if child.is_dir():
                continue
            rel = child.relative_to(root).as_posix()
            if rel == "SKILL.md":
                continue
            files.append(rel)
            if len(files) >= max_files:
                break
        return files


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content.strip()
    lines = content.splitlines()
    end = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break
    if end <= 0:
        return {}, content.strip()

    raw = lines[1:end]
    body = "\n".join(lines[end + 1:]).strip()
    meta: dict[str, str] = {}
    i = 0
    while i < len(raw):
        line = raw[i]
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value in {"|", ">"} or (
            value == ""
            and i + 1 < len(raw)
            and bool(raw[i + 1])
            and raw[i + 1].startswith((" ", "\t"))
        ):
            i += 1
            block: list[str] = []
            while i < len(raw):
                nxt = raw[i]
                if nxt and not nxt.startswith((" ", "\t")) and re.match(r"^[A-Za-z_][\w-]*:\s*", nxt):
                    break
                block.append(nxt.lstrip())
                i += 1
            meta[key] = "\n".join(block).strip()
            continue
        meta[key] = value.strip().strip('"\'')
        i += 1
    return meta, body


def _parse_listish(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    raw = raw.strip("[]")
    parts = re.split(r"[,，、;；/／\n|]+", raw)
    return [p.strip().strip('"\'') for p in parts if p.strip()]


def _parse_state_bias(raw: str) -> dict[str, float]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    bias: dict[str, float] = {}
    for part in re.split(r"[,;；\n]+", raw):
        chunk = part.strip()
        if not chunk:
            continue
        match = re.match(r"^([A-Za-z_][\w-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)$", chunk)
        if not match:
            continue
        bias[match.group(1)] = float(match.group(2))
    return bias


_STATE_RULE_RE = re.compile(
    r"^([A-Za-z_][\w-]*)(?:\s*(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?))?\s*(?:=>|:=)\s*(-?\d+(?:\.\d+)?)(?:\s+if\s+(.+))?$",
    re.IGNORECASE,
)
_STATE_CONDITION_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?)$")
_STATE_BARE_CONDITION_RE = re.compile(r"^([A-Za-z_][\w-]*)$")
_STATE_CONDITION_SPLIT_RE = re.compile(r"\s+(?:and|&&)\s+", re.IGNORECASE)


def _parse_state_condition(raw: str) -> SkillStateCondition | None:
    chunk = (raw or "").strip()
    if not chunk:
        return None
    match = _STATE_CONDITION_RE.match(chunk)
    if match:
        return SkillStateCondition(
            signal_name=match.group(1),
            operator="==" if match.group(2) == "=" else match.group(2),
            threshold=float(match.group(3)),
        )
    bare = _STATE_BARE_CONDITION_RE.match(chunk)
    if bare:
        return SkillStateCondition(signal_name=bare.group(1), operator=">=", threshold=0.5)
    return None


def _parse_state_conditions(raw: str) -> list[SkillStateCondition]:
    text = (raw or "").strip()
    if not text:
        return []
    conditions: list[SkillStateCondition] = []
    for part in _STATE_CONDITION_SPLIT_RE.split(text):
        condition = _parse_state_condition(part)
        if condition is not None:
            conditions.append(condition)
    return conditions


def _parse_state_rules(raw: str) -> list[SkillStateRule]:
    text = (raw or "").strip()
    if not text:
        return []
    rules: list[SkillStateRule] = []
    for part in re.split(r"[,;；\n]+", text):
        chunk = part.strip()
        if not chunk:
            continue
        inhibit_match = re.match(r"^inhibit(?:\s+if)?\s+(.+)$", chunk, flags=re.IGNORECASE)
        if inhibit_match:
            conditions = _parse_state_conditions(inhibit_match.group(1))
            if conditions:
                rules.append(SkillStateRule(inhibit=True, conditions=conditions))
            continue
        legacy_match = re.match(r"^([A-Za-z_][\w-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)$", chunk)
        if legacy_match:
            rules.append(SkillStateRule(
                signal_name=legacy_match.group(1),
                weight=float(legacy_match.group(2)),
            ))
            continue
        match = _STATE_RULE_RE.match(chunk)
        if not match:
            continue
        signal_name = match.group(1)
        conditions: list[SkillStateCondition] = []
        if match.group(2) and match.group(3):
            conditions.append(SkillStateCondition(
                signal_name=signal_name,
                operator="==" if match.group(2) == "=" else match.group(2),
                threshold=float(match.group(3)),
            ))
        if match.group(5):
            conditions.extend(_parse_state_conditions(match.group(5)))
        rules.append(SkillStateRule(
            signal_name=signal_name,
            weight=float(match.group(4)),
            conditions=conditions,
        ))
    return rules


def _parse_metadata_map(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    data: dict[str, str] = {}
    for line in raw.splitlines():
        chunk = line.strip()
        if not chunk or chunk.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][\w.-]*):\s*(.*)$", chunk)
        if not match:
            continue
        data[match.group(1)] = match.group(2).strip().strip('"\'')
    return data


def _parse_allowed_tools(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split() if item.strip()]


_STANDARD_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _warn_skill_shape(name: str, description: str, md_file: Path) -> None:
    if len(name) > 64 or not _STANDARD_SKILL_NAME_RE.fullmatch(name):
        _log.warning("[skill] %s name=%r 不符合 Agent Skills 推荐约束，已按宽松模式加载", md_file, name)
    if md_file.name == "SKILL.md" and md_file.parent.name != name:
        _log.warning("[skill] %s 的 name=%r 与父目录 %r 不一致，已按宽松模式加载", md_file, name, md_file.parent.name)
    if not description.strip():
        _log.warning("[skill] %s description 为空；该 skill 将难以被 catalog 触发", md_file)


def _extract_trigger_text(description: str, meta: dict[str, str]) -> list[str]:
    triggers: list[str] = []
    for key in ("trigger", "triggers"):
        if key in meta:
            triggers.extend(_parse_listish(meta[key]))
    m = re.search(r"(?:Triggers?|触发(?:词|器|条件)?)[：:]\s*(.+)$", description, flags=re.IGNORECASE | re.DOTALL)
    if m:
        triggers.extend(_parse_listish(m.group(1)))
    return [t for t in dict.fromkeys(t.strip() for t in triggers if t.strip())]


def _extract_match_terms(meta: dict[str, str]) -> list[str]:
    terms: list[str] = []
    for key in ("match_terms", "matchers", "anchors", "context_terms"):
        if key in meta:
            terms.extend(_parse_listish(meta[key]))
    return [t for t in dict.fromkeys(t.strip() for t in terms if t.strip())]


_MATCH_RULE_RE = re.compile(
    r"^(contains|any|all)\s*:\s*(.+?)(?:\s*(?:=>|:=)\s*(-?\d+(?:\.\d+)?))?$",
    re.IGNORECASE,
)
_MATCH_RULE_INHIBIT_RE = re.compile(r"^inhibit(?:\s+if)?\s+(contains|any|all)\s*:\s*(.+)$", re.IGNORECASE)


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _parse_match_rule_terms(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*(?:\||,|，|;|；|/|／)\s*", text)
    return [term for term in (_normalize_match_text(part) for part in parts) if term]


def _parse_match_rules(raw: str) -> list[SkillMatchRule]:
    text = (raw or "").strip()
    if not text:
        return []
    rules: list[SkillMatchRule] = []
    for part in re.split(r"[\n;；]+", text):
        chunk = part.strip()
        if not chunk:
            continue
        inhibit = _MATCH_RULE_INHIBIT_RE.match(chunk)
        if inhibit:
            terms = _parse_match_rule_terms(inhibit.group(2))
            if terms:
                rules.append(SkillMatchRule(mode=inhibit.group(1).lower(), terms=terms, inhibit=True))
            continue
        match = _MATCH_RULE_RE.match(chunk)
        if not match:
            continue
        terms = _parse_match_rule_terms(match.group(2))
        if not terms:
            continue
        rules.append(SkillMatchRule(
            mode=match.group(1).lower(),
            terms=terms,
            weight=float(match.group(3) or 1.0),
        ))
    return rules


def _legacy_match_rules(match_terms: list[str], triggers: list[str]) -> list[SkillMatchRule]:
    rules: list[SkillMatchRule] = []
    for term in match_terms:
        normalized = _normalize_match_text(term)
        if normalized:
            rules.append(SkillMatchRule(mode="contains", terms=[normalized], weight=1.0))
    for term in triggers:
        normalized = _normalize_match_text(term)
        if normalized:
            rules.append(SkillMatchRule(mode="contains", terms=[normalized], weight=0.7))
    return rules


def _trim_guidance(text: str, limit: int = 1600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    if "\n" in trimmed:
        trimmed = trimmed.rsplit("\n", 1)[0]
    return trimmed.rstrip() + "\n\n[技能内容已截断，保留前段核心 guidance]"


def _state_signal_values(
    *,
    has_active_task: bool,
    has_next_step: bool,
    failure_count: int,
    high_error_streak: int,
    wm_pressure: float,
    failure_threshold: int,
    wm_pressure_threshold: float,
) -> dict[str, float]:
    failure_ratio = min(1.0, failure_count / max(1, failure_threshold))
    error_denominator = max(1, failure_threshold - 1)
    error_ratio = min(1.0, high_error_streak / error_denominator)
    failure_signal_ratio = max(failure_ratio, error_ratio)
    wm_floor = max(0.2, wm_pressure_threshold)
    wm_pressure_ratio = min(1.0, wm_pressure / wm_floor) if wm_floor > 0 else 0.0
    return {
        "idle_only": 1.0 if not has_active_task and not has_next_step else 0.0,
        "has_active_task": 1.0 if has_active_task else 0.0,
        "has_next_step": 1.0 if has_next_step else 0.0,
        "failure_signal": 1.0 if failure_signal_ratio > 0 else 0.0,
        "failure_signal_ratio": failure_signal_ratio,
        "failure_count_ratio": failure_ratio,
        "high_error_ratio": error_ratio,
        "wm_pressure": 1.0 if wm_pressure >= wm_floor else 0.0,
        "wm_pressure_ratio": wm_pressure_ratio,
    }


def _state_score(
    skill: Skill,
    *,
    state_values: dict[str, float],
) -> float:
    rules = list(skill.state_rules or [])
    if not rules:
        rules = [
            SkillStateRule(signal_name=signal_name, weight=weight)
            for signal_name, weight in (skill.state_bias or {}).items()
        ]
    total = 0.0
    for rule in rules:
        if not all(_state_condition_matches(condition, state_values) for condition in (rule.conditions or [])):
            continue
        if rule.inhibit:
            return float("-inf")
        signal_value = state_values.get(rule.signal_name, 0.0) if rule.signal_name else 1.0
        total += signal_value * rule.weight
    return total


def _state_condition_matches(condition: SkillStateCondition, state_values: dict[str, float]) -> bool:
    current = state_values.get(condition.signal_name, 0.0)
    expected = condition.threshold
    if condition.operator == ">=":
        return current >= expected
    if condition.operator == "<=":
        return current <= expected
    if condition.operator == ">":
        return current > expected
    if condition.operator == "<":
        return current < expected
    return abs(current - expected) <= 1e-9


def _context_score(skill: Skill, context_text: str) -> float:
    hay = _normalize_match_text(context_text)
    if not hay:
        return 0.0

    score = 0.0
    for rule in skill.match_rules or []:
        if not _match_rule_matches(rule, hay):
            continue
        if rule.inhibit:
            return float("-inf")
        score += rule.weight
    return score


def _match_rule_matches(rule: SkillMatchRule, hay: str) -> bool:
    terms = [term for term in rule.terms if term]
    if not terms:
        return False
    if rule.mode == "all":
        return all(term in hay for term in terms)
    return any(term in hay for term in terms)


def _skill_activation_text(skill: Skill, *, include_frontmatter: bool = False, guidance_limit: int | None = None) -> str:
    content = skill.load_markdown()
    meta, body = _split_frontmatter(content)
    guidance = (content if include_frontmatter else (body or content)).strip()
    if guidance_limit is not None and guidance_limit > 0:
        guidance = _trim_guidance(guidance, limit=guidance_limit)
    resources = skill.list_resources()
    lines = [
        f"<skill_content name=\"{skill.name}\">",
        guidance or "（该 skill 目前只有 metadata，没有额外 instructions）",
        "",
        f"Skill directory: {skill.skill_dir}",
        f"Skill source: {skill.source_path}",
    ]
    if skill.compatibility:
        lines.append(f"Compatibility: {skill.compatibility}")
    if skill.allowed_tools:
        lines.append(f"Allowed tools: {' '.join(skill.allowed_tools)}")
    if resources:
        lines.append("<skill_resources>")
        for rel in resources:
            lines.append(f"- {rel}")
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)


# ── 技能注册表 ────────────────────────────────────────────────────────────────

class SkillRegistry:
    """技能注册表：内置技能 + workspace 自定义技能。"""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills: list[Skill] = []
        workspace_loaded = 0
        if skills_dir is not None:
            loaded = self._load_from_dir(skills_dir, origin="workspace")
            workspace_loaded = loaded
            if loaded:
                _log.info("[skill] 从 %s 加载了 %d 个自定义技能", skills_dir, loaded)
        if workspace_loaded <= 0:
            seed_dir = _seed_skills_dir()
            seed_loaded = self._load_from_dir(seed_dir, origin="seed")
            if seed_loaded <= 0:
                _log.warning("[skill] 未从 %s 加载到任何 seed skills", seed_dir)

    def _iter_skill_files(self, skills_dir: Path) -> list[Path]:
        return _iter_skill_files(skills_dir)

    def _load_from_dir(self, skills_dir: Path, *, origin: str) -> int:
        if not skills_dir.exists():
            return 0
        loaded = 0
        for md_file in self._iter_skill_files(skills_dir):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                meta, _body = _split_frontmatter(content)
                name = meta.get("name") or (md_file.parent.name if md_file.name == "SKILL.md" else md_file.stem)
                description = meta.get("description") or f"自定义技能: {name}"
                _warn_skill_shape(name, description, md_file)
                tags = _parse_listish(meta.get("tags", "")) or ["custom"]
                triggers = _extract_trigger_text(description, meta)
                match_terms = _extract_match_terms(meta)
                match_rules = _parse_match_rules(meta.get("match_rules", "")) or _legacy_match_rules(match_terms, triggers)
                raw_state_bias = meta.get("state_bias", "")
                raw_state_rules = meta.get("state_rules", "") or raw_state_bias
                state_bias = _parse_state_bias(raw_state_bias)
                state_rules = _parse_state_rules(raw_state_rules)
                aliases = _parse_listish(meta.get("aliases", ""))
                skill = Skill(
                    name=name,
                    description=description,
                    guidance="",
                    tags=tags,
                    triggers=triggers,
                    match_terms=match_terms,
                    match_rules=match_rules,
                    state_bias=state_bias,
                    state_rules=state_rules,
                    aliases=aliases,
                    compatibility=str(meta.get("compatibility") or "").strip(),
                    license=str(meta.get("license") or "").strip(),
                    allowed_tools=_parse_allowed_tools(meta.get("allowed-tools", "") or meta.get("allowed_tools", "")),
                    metadata=_parse_metadata_map(meta.get("metadata", "")),
                    origin=origin,
                    source_path=str(md_file),
                )
                existing = next((i for i, s in enumerate(self._skills) if s.name == name), -1)
                if existing >= 0:
                    previous = self._skills[existing]
                    self._skills[existing] = skill
                    _log.debug("[skill] %s 覆盖 %s: %s", origin, previous.origin, name)
                else:
                    self._skills.append(skill)
                loaded += 1
            except Exception as exc:
                _log.warning("[skill] 加载 %s 失败: %s", md_file, exc)
        return loaded

    def all_skills(self) -> list[Skill]:
        return list(self._skills)

    def get(self, name: str) -> Skill | None:
        # 1. 精确匹配
        for skill in self._skills:
            if skill.name == name:
                return skill
        # 2. alias 查找：dotted 历史名 → canonical，或匹配 skill.aliases 列表
        canonical = _canonical_skill_name(name)
        for skill in self._skills:
            if skill.name == canonical or name in (skill.aliases or []):
                return skill
        return None

    def activate(self, name: str, *, include_frontmatter: bool = False, guidance_limit: int | None = None) -> tuple[Skill | None, str]:
        skill = self.get(name)
        if skill is None:
            return None, ""
        text = _skill_activation_text(
            skill,
            include_frontmatter=include_frontmatter,
            guidance_limit=guidance_limit,
        )
        return skill, text

    def match_for_context(
        self,
        last_applied: list[str] | None = None,
        max_inject: int = 0,
        **_kwargs: Any,
    ) -> list[Skill]:
        """返回本轮应提示给 LLM 的候选技能列表。

        last_applied: 上轮 LLM 实际应用的技能名列表，优先保留（LLM 自己的选择驱动下轮 activation）。
        max_inject: 最多提示多少个候选技能；0 = 不限（向后兼容）。
        """
        all_skills = list(self._skills)
        context_text = str(_kwargs.get("context_text") or "")
        has_active_task = bool(_kwargs.get("has_active_task"))
        has_next_step = bool(_kwargs.get("has_next_step"))
        failure_count = int(_kwargs.get("failure_count") or 0)
        high_error_streak = int(_kwargs.get("high_error_streak") or 0)
        wm_pressure = float(_kwargs.get("wm_pressure") or 0.0)
        failure_threshold = max(1, int(_kwargs.get("failure_threshold") or 3))
        wm_pressure_threshold = float(_kwargs.get("wm_pressure_threshold") or 0.4)
        applied_names = set(last_applied or [])
        state_values = _state_signal_values(
            has_active_task=has_active_task,
            has_next_step=has_next_step,
            failure_count=failure_count,
            high_error_streak=high_error_streak,
            wm_pressure=wm_pressure,
            failure_threshold=failure_threshold,
            wm_pressure_threshold=wm_pressure_threshold,
        )

        scored: list[tuple[float, int, Skill]] = []
        for index, skill in enumerate(all_skills):
            score = _state_score(skill, state_values=state_values)
            score += _context_score(skill, context_text)
            if skill.name in applied_names and score > 0:
                score += 0.35
            if score > 0:
                scored.append((score, index, skill))

        scored.sort(key=lambda item: (-item[0], item[1]))
        ordered = [skill for _, _, skill in scored]

        if max_inject <= 0:
            selected = ordered + [skill for skill in all_skills if skill not in ordered]
        elif ordered:
            selected = ordered[:max_inject]
        else:
            selected = [skill for skill in all_skills if skill.name in applied_names][:max_inject]

        top_scores = ", ".join(
            f"{skill.name}={score:.2f}" for score, _, skill in scored[:max(3, max_inject or 3)]
        ) or "none"
        _log.info(
            "[skill.match] selected=%d/%d (max=%d last_applied=%s scores=%s): %s",
            len(selected), len(all_skills), max_inject,
            list(last_applied or []),
            top_scores,
            [s.name for s in selected],
        )
        return selected
