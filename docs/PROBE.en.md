# Probe System

[中文](PROBE.md) | [English](PROBE.en.md)

The probe system is an LLM-deployed sensing network. The model can install, remove, trigger, and review probes that pull external signals such as process status, service health, file changes, or HTTP responses back into its cognitive context.

## 1. Probe Kinds

| kind | `spec` content | Meaning |
|------|----------------|---------|
| `shell` | shell command | stdout becomes the probe result |
| `http` | URL | HTTP GET body becomes the result |
| `python` | Python snippet | stdout from executed code becomes the result |

## 2. Trigger Modes

| trigger | Meaning |
|---------|---------|
| `interval:<seconds>` | automatically runs every N seconds |
| `manual` | only runs when `probe.run` is called |

## 3. Result Delivery

| data_back | Meaning |
|-----------|---------|
| `wm` | write probe result into working memory |
| `none` | keep it in logs only |

Manual probe runs always return directly to the caller regardless of `data_back`.

## 4. Alert Expressions

You can attach a Python boolean expression to a probe. The variable `output` contains the result text.

```text
alert_expr:    "float(output.strip()) > 85.0"
alert_message: "CPU temperature too high: {output}"
```

When triggered, the alert message is injected into working memory.

## 5. `purpose` Is Mandatory

The `purpose` field explains why the probe exists, how to interpret its values, and what action is expected when the reading is abnormal. This field is crucial because later model turns need to understand not only the reading, but also the operational meaning behind the sensor.

Good example:

```text
purpose: Monitor Redis memory usage; if it exceeds 500MB, task caching is likely piling up and the scheduler should be checked.
```

Bad example:

```text
purpose: Monitor Redis
```

Blind-spot coverage no longer inspects free-form `purpose/spec` text. It now depends only on explicit `coverage_tags`.

Recommended coverage tags:
- `ops:channel_health`: external channel / proxy / API gateway health
- `ops:api_quota`: API quota, rate limit, or credit usage
- `workspace:git_state`: git changes and workspace state

If `coverage_tags` is omitted, the probe still runs, but it does not count toward blind-spot coverage.

## 6. Available Probe Tools

- `probe.install`
- `probe.run`
- `probe.list`
- `probe.disable`
- `probe.enable`
- `probe.remove`

`probe.install` also accepts optional `coverage_tags`, for example `['ops:channel_health']`.

## 7. Probe Panel in Judgment Context

Every judgment turn can include a sensor panel that shows:

- whether the probe is enabled
- its purpose
- its latest result or last error
- whether an alert fired

That gives the model a persistent operational view of its sensing network.

## 8. Storage

Probe definitions are persisted in `{workspace_dir}/probes.json`, deliberately decoupled from the main SQLite runtime database.

Benefits:

- easy to inspect with a text editor
- independent from DB schema changes
- full restoration after restart, including latest readings

## 9. Common Use Cases

- monitor system resource pressure
- check HTTP service health
- track error log growth
- compute custom metrics with Python snippets