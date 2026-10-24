import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
file = ROOT / "eval" / "agent_questions.json"
API_URL = (
    os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
    + "/agent-graph"
)

cases = json.loads(file.read_text(encoding="utf-8"))
passed = 0

for i, case in enumerate(cases, 1):
    body = json.dumps({"message": case["message"]}).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    actual = result.get("intent")
    expected = case["expected_intent"]
    ok = actual == expected

    if ok:
        passed += 1

    print(f"{i}. {case['message']}")
    print(f"   预期: {expected}，实际: {actual}")
    print(f"   结果: {'通过' if ok else '失败'}")

print(f"\n总结果: {passed}/{len(cases)}")
print(f"路由准确率: {passed / len(cases):.0%}")
