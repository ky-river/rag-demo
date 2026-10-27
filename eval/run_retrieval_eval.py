import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    from eval.metrics import (
        format_ratio,
        latency_summary,
        utc_timestamp,
        write_reports,
    )
except ModuleNotFoundError:
    from metrics import format_ratio, latency_summary, utc_timestamp, write_reports


ROOT = Path(__file__).resolve().parents[1]
SUITE_FILES = {
    "smoke": ROOT / "eval" / "retrieval_questions.json",
    "extended": ROOT / "eval" / "retrieval_questions_extended.json",
}

parser = argparse.ArgumentParser(description="Run retrieval evaluation suites.")
parser.add_argument(
    "--suite",
    choices=sorted(SUITE_FILES),
    default="smoke",
    help="Evaluation suite to run.",
)
parser.add_argument(
    "--api-url",
    default="http://127.0.0.1:8000/search",
    help="Search endpoint to evaluate.",
)
parser.add_argument(
    "--report-dir",
    default=str(ROOT / "eval" / "reports"),
    help="Directory for JSON and Markdown reports.",
)
parser.add_argument(
    "--no-report",
    action="store_true",
    help="Print results without writing report files.",
)
args = parser.parse_args()

API_TOKEN = os.getenv("RAG_API_TOKEN", "").strip()
CASES_FILE = SUITE_FILES[args.suite]
API_URL = args.api_url
cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
answerable = [case for case in cases if not case.get("expect_no_sources")]
refusal = [case for case in cases if case.get("expect_no_sources")]

hit_at_1 = 0
hit_at_4 = 0
reciprocal_rank = 0.0
refusal_correct = 0
errors = 0
latencies = []
case_results = []
failures = []
category_metrics = defaultdict(
    lambda: {"hit_at_1": 0, "hit_at_4": 0, "reciprocal_rank": 0.0, "count": 0}
)
category_refusals = defaultdict(lambda: {"correct": 0, "count": 0})


def matching_rank(sources, expected_sources):
    for rank, source in enumerate(sources, 1):
        for expected in expected_sources:
            if source.get("filename") != expected["filename"]:
                continue
            content = source.get("content", "")
            if any(keyword in content for keyword in expected["any_keywords"]):
                return rank
    return None


for index, case in enumerate(cases, 1):
    body = json.dumps({"question": case["question"]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers=headers,
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        errors += 1
        request_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        latencies.append(request_latency_ms)
        reason = f"请求失败：{exc}"
        failures.append({"question": case["question"], "reason": reason})
        case_results.append(
            {
                "question": case["question"],
                "category": case.get("category", "answerable"),
                "status": "error",
                "reason": reason,
                "latency_ms": request_latency_ms,
            }
        )
        print(f"{index}. {case['question']}")
        print(f"   {reason}")
        continue

    if result.get("error"):
        errors += 1
        reason = f"接口返回错误：{result['error']}"
        failures.append({"question": case["question"], "reason": reason})
        case_results.append(
            {
                "question": case["question"],
                "category": case.get("category", "answerable"),
                "status": "error",
                "reason": reason,
                "latency_ms": result.get("latency_ms", 0),
            }
        )
        print(f"{index}. {case['question']}")
        print(f"   {reason}")
        continue

    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    server_latency = result.get("latency_ms")
    latency_ms = (
        float(server_latency)
        if server_latency is not None
        else round((time.perf_counter() - started_at) * 1000, 2)
    )
    latencies.append(latency_ms)

    if case.get("expect_no_sources"):
        passed = not sources
        refusal_correct += passed
        category = case.get("category", "refusal")
        category_refusals[category]["correct"] += passed
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
                "question": case["question"],
                "category": category,
                "status": "passed" if passed else "failed",
                "source_count": len(sources),
                "latency_ms": round(latency_ms, 2),
            }
        )
        print(f"{index}. {case['question']}")
        print(f"   来源数量: {len(sources)}，拒答: {'通过' if passed else '失败'}")
        continue

    rank = matching_rank(sources, case["expected_sources"])
    passed = rank is not None and rank <= 4
    hit_at_1 += rank == 1
    hit_at_4 += passed
    reciprocal_rank += 1 / rank if rank else 0
    category = case.get("category", "answerable")
    category_metrics[category]["count"] += 1
    category_metrics[category]["hit_at_1"] += rank == 1
    category_metrics[category]["hit_at_4"] += passed
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
            "question": case["question"],
            "category": category,
            "status": "passed" if passed else "failed",
            "rank": rank,
            "returned_filenames": [
                source.get("filename", "") for source in sources
            ],
            "latency_ms": round(latency_ms, 2),
        }
    )

    print(f"{index}. {case['question']}")
    print(f"   正确来源排名: {rank if rank else '未命中'}，结果: {'通过' if passed else '失败'}")

mrr = reciprocal_rank / len(answerable) if answerable else 0.0
latency_stats = latency_summary(latencies)
report_status = (
    "passed"
    if hit_at_4 == len(answerable)
    and refusal_correct == len(refusal)
    and errors == 0
    else "failed"
)

print(f"\n评测套件: {args.suite}")
print(f"有答案问题: {len(answerable)}")
print(f"Hit@1: {hit_at_1}/{len(answerable)}")
print(f"Hit@4: {hit_at_4}/{len(answerable)}")
print(f"MRR: {mrr:.3f}")
print(f"无答案拒答正确: {refusal_correct}/{len(refusal)}")
print(f"平均检索延迟: {latency_stats['average_ms']:.2f} ms")
print(f"P95 检索延迟: {latency_stats['p95_ms']:.2f} ms")
if errors:
    print(f"请求错误: {errors}")

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

if category_rows:
    print("\n分类结果:")
    for category in category_rows:
        values = "；".join(
            f"{metric['name']} {metric['value']}"
            for metric in category["metrics"]
        )
        print(f"  {category['name']}: {values}")

report = {
    "title": "RAG 检索评测报告",
    "kind": "retrieval",
    "status": report_status,
    "generated_at": utc_timestamp(),
    "meta": {
        "评测套件": args.suite,
        "接口": API_URL,
        "测试用例": str(CASES_FILE.relative_to(ROOT)),
    },
    "metrics": [
        {"name": "问题总数", "value": str(len(cases))},
        {"name": "Hit@1", "value": format_ratio(hit_at_1, len(answerable))},
        {"name": "Hit@4", "value": format_ratio(hit_at_4, len(answerable))},
        {"name": "MRR", "value": f"{mrr:.3f}"},
        {
            "name": "无答案拒答",
            "value": format_ratio(refusal_correct, len(refusal)),
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

if not args.no_report:
    json_path, markdown_path = write_reports(
        report,
        args.report_dir,
        f"retrieval-{args.suite}",
    )
    print(f"\nJSON 报告: {json_path}")
    print(f"Markdown 报告: {markdown_path}")

if report_status != "passed":
    raise SystemExit(1)
