from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def percentile(values: Iterable[float], percent: float) -> float:
    """Return a linearly interpolated percentile for a small local sample."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 2)

    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)

    weight = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * weight
    return round(value, 2)


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    latencies = [float(value) for value in values]
    if not latencies:
        return {
            "count": 0,
            "average_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": len(latencies),
        "average_ms": round(sum(latencies) / len(latencies), 2),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "max_ms": round(max(latencies), 2),
    }


def format_ratio(passed: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0.0%)"
    return f"{passed}/{total} ({passed / total:.1%})"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_markdown(report: dict[str, Any]) -> str:
    status_label = "通过" if report.get("status") == "passed" else "未通过"
    lines = [
        f"# {report.get('title', '评测报告')}",
        "",
        f"- 状态：{status_label}",
        f"- 生成时间：{report.get('generated_at', '-')}",
        "",
        "## 运行信息",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
    ]
    for key, value in report.get("meta", {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## 核心指标",
            "",
            "| 指标 | 结果 |",
            "| --- | --- |",
        ]
    )
    for metric in report.get("metrics", []):
        lines.append(f"| {metric['name']} | {metric['value']} |")

    categories = report.get("categories", [])
    if categories:
        lines.extend(["", "## 分类指标"])
        for category in categories:
            lines.extend(
                [
                    "",
                    f"### {category['name']}",
                    "",
                    "| 指标 | 结果 |",
                    "| --- | --- |",
                ]
            )
            for metric in category.get("metrics", []):
                lines.append(f"| {metric['name']} | {metric['value']} |")

    failures = report.get("failures", [])
    lines.extend(["", "## 失败明细", ""])
    if failures:
        for index, failure in enumerate(failures, 1):
            lines.append(
                f"{index}. {failure.get('question', '未知问题')}："
                f"{failure.get('reason', '未提供原因')}"
            )
    else:
        lines.append("无。")

    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    report: dict[str, Any],
    output_dir: str | Path,
    filename_prefix: str,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", filename_prefix).strip("-")
    base_name = f"{safe_prefix or 'evaluation'}-{timestamp}"
    json_path = destination / f"{base_name}.json"
    markdown_path = destination / f"{base_name}.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
