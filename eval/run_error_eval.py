import json
import os
import urllib.request


BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("RAG_API_TOKEN", "").strip()


def authorization_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def post_json(path, payload):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={
            "Content-Type": "application/json",
            **authorization_headers(),
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_file(path, filename, content):
    boundary = "----RagErrorEvalBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **authorization_headers(),
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


tests = [
    ("空 Agent 任务", lambda: post_json("/agent", {"message": ""}), "请输入任务"),
    ("空知识库问题", lambda: post_json("/ask", {"question": ""}), "问题不能为空"),
    ("无技能缺口学习计划", lambda: post_json("/study-plan", {"gaps": []}), "没有可用于制定计划的技能缺口"),
    ("非 PDF 上传", lambda: post_file("/upload", "notes.txt", b"not a pdf"), "只支持 PDF 文件"),
]

passed = 0
for name, call, expected in tests:
    status, result = call()
    text = json.dumps(result, ensure_ascii=False)
    ok = expected in text
    passed += ok
    print(f"{name}: {'通过' if ok else '失败'} (HTTP {status})")
    if not ok:
        print(f"  返回: {text}")

print(f"\n异常测试结果: {passed}/{len(tests)}")
print(f"异常测试通过率: {passed / len(tests):.0%}")
