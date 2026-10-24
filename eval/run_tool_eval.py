import json
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "eval" / "tool_questions.json"
API_URL = (
    os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
    + "/agent-graph"
)

cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
passed = 0
model_selected = 0

for index, case in enumerate(cases, 1):
    body = json.dumps({"message": case["message"]}).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    actual_tool = result.get("tool")
    expected_tool = case["expected_tool"]
    routing = result.get("routing", "unknown")
    ok = actual_tool == expected_tool

    if ok:
        passed += 1
    if routing == "model":
        model_selected += 1

    print(f"{index}. {case['message']}")
    print(f"   预期工具: {expected_tool}，实际工具: {actual_tool}")
    print(f"   路由来源: {routing}，结果: {'通过' if ok else '失败'}")

print(f"\n工具选择正确: {passed}/{len(cases)}")
print(f"工具选择准确率: {passed / len(cases):.0%}")
print(f"模型主动选择工具: {model_selected}/{len(cases)}")

if passed != len(cases):
    raise SystemExit(1)
