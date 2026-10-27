import argparse
import json
import os
import time
import urllib.error
import urllib.request
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
QUESTIONS_FILE = ROOT / "eval" / "questions.json"

parser = argparse.ArgumentParser(description="Run answer and citation evaluation.")
parser.add_argument(
    "--api-url",
    default=os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000") + "/ask",
    help="Answer endpoint to evaluate.",
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
questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

correct = 0
citation_correct = 0
errors = 0
latencies = []
case_results = []
failures = []

for index, item in enumerate(questions, start=1):
    payload = json.dumps({"question": item["question"]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    request = urllib.request.Request(
        args.api_url,
        data=payload,
        headers=headers,
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        errors += 1
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        latencies.append(latency_ms)
        reason = f"请求失败：{exc}"
        failures.append({"question": item["question"], "reason": reason})
        case_results.append(
            {
                "question": item["question"],
                "status": "error",
                "reason": reason,
                "latency_ms": latency_ms,
            }
        )
        print(f"\n问题 {index}: {item['question']}")
        print(reason)
        continue

    if result.get("error"):
        errors += 1
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        latencies.append(latency_ms)
        reason = f"接口返回错误：{result['error']}"
        failures.append({"question": item["question"], "reason": reason})
        case_results.append(
            {
                "question": item["question"],
                "status": "error",
                "reason": reason,
                "latency_ms": latency_ms,
            }
        )
        print(f"\n问题 {index}: {item['question']}")
        print(reason)
        continue

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    latencies.append(latency_ms)
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    if item.get("expected_any"):
        passed = any(keyword in answer for keyword in item["expected_any"])
    else:
        keywords = item["expected"].split("、")
        passed = all(keyword.strip() in answer for keyword in keywords)

    if passed:
        correct += 1

    if item.get("expect_no_sources"):
        citation_passed = not sources
    else:
        citation_passed = any(
            source.get("filename") == expected["filename"]
            and any(
                keyword in source.get("content", "")
                for keyword in expected["any_keywords"]
            )
            for source in sources
            for expected in item.get("expected_sources", [])
        )

    if citation_passed:
        citation_correct += 1

    if not passed:
        failures.append(
            {
                "question": item["question"],
                "reason": "回答未包含预期关键词",
            }
        )
    if not citation_passed:
        failures.append(
            {
                "question": item["question"],
                "reason": "引用未命中预期文件、片段或拒答条件",
            }
        )
    case_results.append(
        {
            "question": item["question"],
            "status": "passed" if passed and citation_passed else "failed",
            "answer_passed": passed,
            "citation_passed": citation_passed,
            "answer_preview": answer[:200],
            "source_count": len(sources),
            "latency_ms": latency_ms,
        }
    )

    print(f"\n问题 {index}: {item['question']}")
    print(f"回答: {answer}")
    print(f"来源数量: {len(sources)}")
    print(f"回答结果: {'通过' if passed else '未通过'}")
    print(f"引用结果: {'通过' if citation_passed else '未通过'}")

latency_stats = latency_summary(latencies)
report_status = (
    "passed"
    if correct == len(questions)
    and citation_correct == len(questions)
    and errors == 0
    else "failed"
)

print(f"\n总结果: {correct}/{len(questions)}")
print(f"简单匹配准确率: {correct / len(questions):.0%}")
print(f"引用正确: {citation_correct}/{len(questions)}")
print(f"引用正确率: {citation_correct / len(questions):.0%}")
print(f"平均端到端延迟: {latency_stats['average_ms']:.2f} ms")
print(f"P95 端到端延迟: {latency_stats['p95_ms']:.2f} ms")
if errors:
    print(f"请求错误: {errors}")

report = {
    "title": "RAG 回答与引用评测报告",
    "kind": "answer",
    "status": report_status,
    "generated_at": utc_timestamp(),
    "meta": {
        "接口": args.api_url,
        "测试用例": str(QUESTIONS_FILE.relative_to(ROOT)),
    },
    "metrics": [
        {"name": "问题总数", "value": str(len(questions))},
        {
            "name": "回答通过",
            "value": format_ratio(correct, len(questions)),
        },
        {
            "name": "引用通过",
            "value": format_ratio(citation_correct, len(questions)),
        },
        {
            "name": "平均端到端延迟",
            "value": f"{latency_stats['average_ms']:.2f} ms",
        },
        {
            "name": "P50 端到端延迟",
            "value": f"{latency_stats['p50_ms']:.2f} ms",
        },
        {
            "name": "P95 端到端延迟",
            "value": f"{latency_stats['p95_ms']:.2f} ms",
        },
        {"name": "请求错误", "value": str(errors)},
    ],
    "categories": [],
    "failures": failures,
    "cases": case_results,
}

if not args.no_report:
    json_path, markdown_path = write_reports(
        report,
        args.report_dir,
        "answer-citation",
    )
    print(f"\nJSON 报告: {json_path}")
    print(f"Markdown 报告: {markdown_path}")

if report_status != "passed":
    raise SystemExit(0)