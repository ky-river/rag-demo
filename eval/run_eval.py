import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
questions_file = ROOT / "eval" / "questions.json"
api_url = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000") + "/ask"

questions = json.loads(questions_file.read_text(encoding="utf-8"))

correct = 0
citation_correct = 0

for index, item in enumerate(questions, start=1):
    payload = json.dumps({"question": item["question"]}).encode("utf-8")

    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    answer = result.get("answer", "")
    sources = result.get("sources", [])
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

    print(f"\n问题 {index}: {item['question']}")
    print(f"回答: {answer}")
    print(f"来源数量: {len(sources)}")
    print(f"回答结果: {'通过' if passed else '未通过'}")
    print(f"引用结果: {'通过' if citation_passed else '未通过'}")

print(f"\n总结果: {correct}/{len(questions)}")
print(f"简单匹配准确率: {correct / len(questions):.0%}")
print(f"引用正确: {citation_correct}/{len(questions)}")
print(f"引用正确率: {citation_correct / len(questions):.0%}")

if correct != len(questions) or citation_correct != len(questions):
    raise SystemExit(1)
