import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError

import app.main as main


client = TestClient(main.app)


class RequestObservabilityTest(unittest.TestCase):
    def test_request_id_is_returned_and_formatter_outputs_json(self):
        health = {
            "status": "ok",
            "version": "test",
            "index_ready": True,
            "document_count": 1,
            "chunk_count": 1,
            "collection_count": 1,
            "collection_error": None,
            "model": {"timeout_seconds": 60, "max_retries": 1},
            "reranking": {"enabled": False},
        }
        with patch.object(main, "health_status", return_value=health):
            response = client.get(
                "/health",
                headers={"X-Request-ID": "request-test-123"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "request-test-123")

        record = logging.LogRecord(
            "rag_demo",
            logging.INFO,
            __file__,
            1,
            "request_completed",
            (),
            None,
        )
        record.event = "request_completed"
        record.request_id = "request-test-123"
        payload = json.loads(main.JsonLogFormatter().format(record))
        self.assertEqual(payload["message"], "request_completed")
        self.assertEqual(payload["event"], "request_completed")
        self.assertEqual(payload["request_id"], "request-test-123")

    def test_model_timeout_returns_504_with_request_id(self):
        timeout_error = APITimeoutError(
            request=httpx.Request("POST", "http://test/models")
        )
        with patch.object(
            main,
            "answer_from_knowledge_base",
            side_effect=timeout_error,
        ):
            response = client.post(
                "/ask",
                headers={"X-Request-ID": "timeout-test-123"},
                json={"question": "测试超时"},
            )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json(),
            {
                "error": "模型请求超时，请稍后重试",
                "request_id": "timeout-test-123",
            },
        )


class ApiEndpointTest(unittest.TestCase):
    def test_ask_returns_grounded_answer_contract(self):
        result = {
            "answer": "Chroma 与 BM25",
            "sources": [{"filename": "project.pdf", "page": 1, "content": "证据"}],
        }
        with patch.object(
            main,
            "answer_from_knowledge_base",
            return_value=result,
        ) as answer:
            response = client.post("/ask", json={"question": " 使用什么检索？ "})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question"], "使用什么检索？")
        self.assertEqual(response.json()["answer"], result["answer"])
        answer.assert_called_once_with("使用什么检索？")

    def test_search_returns_retrieval_metadata(self):
        sources = [{"filename": "project.pdf", "page": 1, "content": "证据"}]
        with (
            patch.object(main, "retrieve_knowledge", return_value=([], sources)),
            patch.object(
                main,
                "reranking_status",
                return_value={"enabled": False, "active": False},
            ),
        ):
            response = client.post("/search", json={"question": "RAG 流程"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sources"], sources)
        self.assertEqual(body["retrieval_mode"], "vector+bm25_rrf")
        self.assertIn("latency_ms", body)

    def test_agent_graph_returns_session_and_routing(self):
        graph_result = {
            "intent": "study_plan",
            "tool": "create_study_plan",
            "routing": "model",
            "answer": "四周计划",
            "sources": [],
        }
        with (
            patch.object(main, "load_history", return_value=[]),
            patch.object(main.agent_graph, "invoke", return_value=graph_result) as invoke,
            patch.object(main, "save_history") as save,
        ):
            response = client.post(
                "/agent-graph",
                json={"message": "制定学习计划", "session_id": "session-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-1")
        self.assertEqual(response.json()["tool"], "create_study_plan")
        invoke.assert_called_once_with(
            {"message": "制定学习计划", "history": []}
        )
        save.assert_called_once()

    def test_clear_agent_session(self):
        with patch.object(main, "clear_history") as clear:
            response = client.post(
                "/agent/clear",
                json={"session_id": "session-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-1")
        clear.assert_called_once_with("session-1")

    def test_study_plan_returns_structured_plan(self):
        plan = {
            "goal": "补齐 RAG 基础",
            "weeks": [
                {
                    "week": 1,
                    "focus": "检索",
                    "tasks": ["实现 BM25"],
                    "deliverable": "检索脚本",
                }
            ],
        }
        model_message = SimpleNamespace(content=json.dumps(plan, ensure_ascii=False))
        model_response = SimpleNamespace(
            choices=[SimpleNamespace(message=model_message)]
        )

        with patch.object(
            main,
            "create_chat_completion",
            return_value=model_response,
        ) as create:
            response = client.post(
                "/study-plan",
                json={"gaps": ["RAG", "LangGraph"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["plan"], plan)
        self.assertEqual(create.call_args.args[0], "study_plan_json")

    def test_resume_match_returns_structured_analysis(self):
        analysis = {
            "match_score": 82,
            "matched_skills": ["Python"],
            "skill_gaps": ["LangGraph"],
            "resume_suggestions": ["补充量化结果"],
            "interview_questions": ["如何评估 RAG？"],
        }
        model_message = SimpleNamespace(
            content=json.dumps(analysis, ensure_ascii=False)
        )
        model_response = SimpleNamespace(
            choices=[SimpleNamespace(message=model_message)]
        )

        with (
            patch.object(
                main,
                "read_uploaded_pdf",
                side_effect=["候选人文本", "岗位文本"],
            ),
            patch.object(
                main,
                "create_chat_completion",
                return_value=model_response,
            ) as create,
        ):
            response = client.post(
                "/match",
                files={
                    "resume": ("resume.pdf", b"%PDF-1.4", "application/pdf"),
                    "job_description": ("job.pdf", b"%PDF-1.4", "application/pdf"),
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysis"], analysis)
        self.assertEqual(create.call_args.args[0], "resume_match")

    def test_upload_rebuilds_index_without_external_model(self):
        fake_page = SimpleNamespace(page_content="可读取的 PDF 文本")

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "DATA_DIR", Path(temp_dir)),
                patch.object(main, "PyPDFLoader") as loader,
                patch.object(main, "build_index", return_value=4) as build,
            ):
                loader.return_value.load.return_value = [fake_page]
                response = client.post(
                    "/upload",
                    files={
                        "file": ("knowledge.pdf", b"%PDF-1.4", "application/pdf")
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunks"], 4)
        build.assert_called_once_with()

    def test_upload_rejects_non_pdf(self):
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"text", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "只支持 PDF 文件"})


if __name__ == "__main__":
    unittest.main()
