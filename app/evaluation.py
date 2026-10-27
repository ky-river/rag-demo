from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from eval.metrics import format_ratio, latency_summary, utc_timestamp


ROOT = Path(__file__).resolve().parents[1]
SUITE_FILES = {
    "smoke": ROOT / "eval" / "retrieval_questions.json",
    "extended": ROOT / "eval" / "retrieval_questions_extended.json",
}


def matching_rank(
    sources: list[dict],
    expected_sources: list[dict],
) -> int | None:
    for rank, source in enumerate(sources, 1):
        for expected in expected_sources:
            if source.get("filename") != expected["filename"]:
                continue
            content = source.get("content", "")
            if any(keyword in content for keyword in expected["any_keywords"]):
                return rank
    return None


def load_cases(suite: str) -> list[dict]:
    try:
        path = SUITE_FILES[suite]
    except KeyError as exc:
        raise ValueError("不支持的评测套件") from exc
    return json.loads(path.read_text(encoding="utf-8"))


def run_retrieval_evaluation(
    suite: str,
    search: Callable[[str], list[dict]],
) -> dict:
    cases = load_cases(suite)
    answerable_count = sum(not case.get("expect_no_sources") for case in cases)
    refusal_count = len(cases) - answerable_count

    hit_at_1 = 0
    hit_at_4 = 0
    reciprocal_rank = 0.0
    refusal_correct = 0
    errors = 0
    latencies: list[float] = []
    failures = []
    case_results = []
    category_metrics = defaultdict(
        lambda: {"hit_at_1": 0, "hit_at_4": 0, "reciprocal_rank": 0.0, "count": 0}
    )
    category_refusals = defaultdict(lambda: {"correct": 0, "count": 0})

    for index, case in enumerate(cases, 1):
        started_at = time.perf_counter()
        try:
            sources = search(case["question"])
            if not isinstance(sources, list):
                raise ValueError("检索结果格式不正确")
        except Exception as exc:
            errors += 1
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            latencies.append(latency_ms)
            reason = f"检索失败：{exc}"
            failures.append({"question": case["question"], "reason": reason})
            case_results.append(
                {
                    "index": index,
                    "question": case["question"],
                    "category": case.get("category", "answerable"),
                    "status": "error",
                    "reason": reason,
                    "latency_ms": latency_ms,
                }
            )
            continue

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        latencies.append(latency_ms)
        category = case.get(
            "category",
            "refusal" if case.get("expect_no_sources") else "answerable",
        )

        if case.get("expect_no_sources"):
            passed = not sources
            refusal_correct += int(passed)
            category_refusals[category]["correct"] += int(passed)
            category_refusals[category]["count"] += 1
            if not passed:
                failures.append(
                    {
                        "question": case["question"],
                        "reason": f"预期拒答，实际返回 {len(sources)} 个来源",
                    }
                )
            case_results.append(
                {
                    "index": index,
                    "question": case["question"],
                    "category": category,
                    "status": "passed" if passed else "failed",
                    "source_count": len(sources),
                    "latency_ms": latency_ms,
                }
            )
            continue

        rank = matching_rank(sources, case["expected_sources"])
        passed = rank is not None and rank <= 4
        hit_at_1 += int(rank == 1)
        hit_at_4 += int(passed)
        reciprocal_rank += 1 / rank if rank else 0
        category_metrics[category]["count"] += 1
        category_metrics[category]["hit_at_1"] += int(rank == 1)
        category_metrics[category]["hit_at_4"] += int(passed)
        category_metrics[category]["reciprocal_rank"] += 1 / rank if rank else 0

        if not passed:
            failures.append(
                {
                    "question": case["question"],
                    "reason": (
                        f"预期来源未进入 Top 4，实际排名：{rank}"
                        if rank
                        else "未在返回来源中命中预期文件和关键词"
                    ),
                }
            )
        case_results.append(
            {
                "index": index,
                "question": case["question"],
                "category": category,
                "status": "passed" if passed else "failed",
                "rank": rank,
                "returned_filenames": [
                    source.get("filename", "") for source in sources
                ],
                "latency_ms": latency_ms,
            }
        )

    mrr = reciprocal_rank / answerable_count if answerable_count else 0.0
    latency_stats = latency_summary(latencies)
    status = (
        "passed"
        if hit_at_4 == answerable_count
        and refusal_correct == refusal_count
        and errors == 0
        else "failed"
    )

    category_rows = []
    for category in sorted(set(category_metrics) | set(category_refusals)):
        metrics = []
        answer_metrics = category_metrics.get(category)
        if answer_metrics and answer_metrics["count"]:
            metrics.extend(
                [
                    {
                        "name": "Hit@1",
                        "value": format_ratio(
                            answer_metrics["hit_at_1"],
                            answer_metrics["count"],
                        ),
                    },
                    {
                        "name": "Hit@4",
                        "value": format_ratio(
                            answer_metrics["hit_at_4"],
                            answer_metrics["count"],
                        ),
                    },
                    {
                        "name": "MRR",
                        "value": (
                            f"{answer_metrics['reciprocal_rank'] / answer_metrics['count']:.3f}"
                        ),
                    },
                ]
            )
        refusal_metrics = category_refusals.get(category)
        if refusal_metrics and refusal_metrics["count"]:
            metrics.append(
                {
                    "name": "拒答",
                    "value": format_ratio(
                        refusal_metrics["correct"],
                        refusal_metrics["count"],
                    ),
                }
            )
        category_rows.append({"name": category, "metrics": metrics})

    return {
        "title": "RAG 检索评测报告",
        "kind": "retrieval",
        "status": status,
        "generated_at": utc_timestamp(),
        "meta": {
            "评测套件": suite,
            "问题数": len(cases),
        },
        "metrics": [
            {"name": "问题总数", "value": str(len(cases))},
            {"name": "Hit@1", "value": format_ratio(hit_at_1, answerable_count)},
            {"name": "Hit@4", "value": format_ratio(hit_at_4, answerable_count)},
            {"name": "MRR", "value": f"{mrr:.3f}"},
            {
                "name": "无答案拒答",
                "value": format_ratio(refusal_correct, refusal_count),
            },
            {
                "name": "平均检索延迟",
                "value": f"{latency_stats['average_ms']:.2f} ms",
            },
            {
                "name": "P50 检索延迟",
                "value": f"{latency_stats['p50_ms']:.2f} ms",
            },
            {
                "name": "P95 检索延迟",
                "value": f"{latency_stats['p95_ms']:.2f} ms",
            },
            {"name": "请求错误", "value": str(errors)},
        ],
        "categories": category_rows,
        "failures": failures,
        "cases": case_results,
    }
