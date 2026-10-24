import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path


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
args = parser.parse_args()

CASES_FILE = SUITE_FILES[args.suite]
API_URL = args.api_url
cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
answerable = [case for case in cases if not case.get("expect_no_sources")]
refusal = [case for case in cases if case.get("expect_no_sources")]

hit_at_1 = 0
hit_at_4 = 0
reciprocal_rank = 0.0
refusal_correct = 0
latencies = []
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
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    sources = result.get("sources", [])
    latencies.append(result.get("latency_ms", 0))

    if case.get("expect_no_sources"):
        passed = not sources
        refusal_correct += passed
        category = case.get("category", "refusal")
        category_refusals[category]["correct"] += passed
        category_refusals[category]["count"] += 1
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

    print(f"{index}. {case['question']}")
    print(f"   正确来源排名: {rank if rank else '未命中'}，结果: {'通过' if passed else '失败'}")

print(f"\n评测套件: {args.suite}")
print(f"有答案问题: {len(answerable)}")
print(f"Hit@1: {hit_at_1}/{len(answerable)}")
print(f"Hit@4: {hit_at_4}/{len(answerable)}")
print(f"MRR: {reciprocal_rank / len(answerable):.3f}")
print(f"无答案拒答正确: {refusal_correct}/{len(refusal)}")
print(f"平均检索延迟: {sum(latencies) / len(latencies):.2f} ms")

if category_metrics or category_refusals:
    print("\n分类结果:")
    for category, metrics in sorted(category_metrics.items()):
        print(
            f"  {category}: Hit@1 {metrics['hit_at_1']}/{metrics['count']}, "
            f"Hit@4 {metrics['hit_at_4']}/{metrics['count']}, "
            f"MRR {metrics['reciprocal_rank'] / metrics['count']:.3f}"
        )
    for category, metrics in sorted(category_refusals.items()):
        print(f"  {category}: 拒答 {metrics['correct']}/{metrics['count']}")

if hit_at_4 != len(answerable) or refusal_correct != len(refusal):
    raise SystemExit(1)
