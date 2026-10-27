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
from app.auth import AuthStore, User


client = TestClient(main.app)
TEST_USER = User(id="test-user", email="test@example.com")
main.app.dependency_overrides[main.require_user] = lambda: TEST_USER


class RequestObservabilityTest(unittest.TestCase):
    def test_request_id_is_returned_and_formatter_outputs_json(self):
        health = {
            "status": "ok",
            "version": "test",
            "index_ready": True,
            "collection_name": "rag_demo",
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
    def setUp(self):
        main.rate_limiter.reset()

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
        answer.assert_called_once_with("使用什么检索？", owner_id=TEST_USER.id)

    def test_search_returns_retrieval_metadata(self):
        sources = [{"filename": "project.pdf", "page": 1, "content": "证据"}]
        with (
            patch.object(main, "retrieve_knowledge", return_value=([], sources)) as retrieve,
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
        retrieve.assert_called_once_with("RAG 流程", TEST_USER.id)

    def test_retrieval_evaluation_uses_current_user_scope(self):
        report = {
            "status": "passed",
            "metrics": [{"name": "Hit@4", "value": "40/40"}],
            "failures": [],
        }
        with (
            patch.object(
                main,
                "run_retrieval_evaluation",
                return_value=report,
            ) as evaluate,
            patch.object(
                main,
                "retrieve_knowledge",
                return_value=([], []),
            ) as retrieve,
        ):
            response = client.post(
                "/eval/retrieval",
                json={"suite": "extended"},
            )
            search = evaluate.call_args.args[1]
            search("测试问题")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "passed")
            self.assertEqual(evaluate.call_args.args[0], "extended")
        retrieve.assert_called_once_with("测试问题", TEST_USER.id)

    def test_agent_graph_returns_session_and_routing(self):
        graph_result = {
            "intent": "study_plan",
            "tool": "create_study_plan",
            "routing": "model",
            "answer": "四周计划",
            "sources": [],
        }
        with (
            patch.object(main, "load_history", return_value=[]) as load,
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
            {
                "owner_id": TEST_USER.id,
                "message": "制定学习计划",
                "history": [],
            }
        )
        load.assert_called_once_with("session-1", TEST_USER.id)
        save.assert_called_once_with(
            "session-1",
            [
                {"role": "user", "content": "制定学习计划"},
                {"role": "assistant", "content": "四周计划"},
            ],
            TEST_USER.id,
        )

    def test_clear_agent_session(self):
        with patch.object(main, "clear_history") as clear:
            response = client.post(
                "/agent/clear",
                json={"session_id": "session-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-1")
        clear.assert_called_once_with("session-1", TEST_USER.id)

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

    def test_documents_returns_file_and_chunk_inventory(self):
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "project.pdf").write_bytes(b"1234567890")
            (data_dir / "RESUME.PDF").write_bytes(b"12345")
            (data_dir / ".backup.pdf").write_bytes(b"hidden")
            (data_dir / "notes.txt").write_text("ignored", encoding="utf-8")
            stored_documents = [
                SimpleNamespace(metadata={"filename": "project.pdf"}),
                SimpleNamespace(metadata={"filename": "project.pdf"}),
                SimpleNamespace(metadata={"filename": "RESUME.PDF"}),
            ]
            workspace = main.WorkspaceIndex(
                collection_name="rag_demo_test",
                vectorstore=SimpleNamespace(),
                bm25_documents=stored_documents,
                bm25_index=None,
            )

            with (
                patch.object(main, "workspace_data_dir", return_value=data_dir),
                patch.object(
                    main,
                    "get_workspace_index_locked",
                    return_value=workspace,
                ),
            ):
                response = client.get("/documents")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["document_count"], 2)
        self.assertEqual(body["chunk_count"], 3)
        self.assertEqual(body["collection"], "rag_demo_test")
        inventory = {item["filename"]: item for item in body["documents"]}
        self.assertEqual(set(inventory), {"project.pdf", "RESUME.PDF"})
        self.assertEqual(inventory["project.pdf"]["size_bytes"], 10)
        self.assertEqual(inventory["project.pdf"]["chunk_count"], 2)
        self.assertEqual(inventory["RESUME.PDF"]["chunk_count"], 1)

    def test_delete_document_rebuilds_index_and_removes_file(self):
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "keep.pdf").write_bytes(b"keep")
            (data_dir / "remove.pdf").write_bytes(b"remove")
            workspace = main.WorkspaceIndex(
                collection_name="rag_demo_build_test",
                vectorstore=SimpleNamespace(),
                bm25_documents=[],
                bm25_index=None,
            )

            with (
                patch.object(main, "workspace_data_dir", return_value=data_dir),
                patch.object(main, "build_index", return_value=5) as build,
                patch.object(
                    main,
                    "get_workspace_index_locked",
                    return_value=workspace,
                ),
            ):
                response = client.delete("/documents/remove.pdf")

            self.assertFalse((data_dir / "remove.pdf").exists())
            self.assertTrue((data_dir / "keep.pdf").exists())
            self.assertEqual(list(data_dir.glob(".deleted_*.pdf")), [])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunks"], 5)
        self.assertEqual(response.json()["collection"], "rag_demo_build_test")
        build.assert_called_once_with(TEST_USER.id)

    def test_delete_document_failure_restores_file_and_old_index(self):
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pdf_path = data_dir / "remove.pdf"
            pdf_path.write_bytes(b"previous-content")
            workspace = main.WorkspaceIndex(
                collection_name="rag_demo_old",
                vectorstore=SimpleNamespace(),
                bm25_documents=[],
                bm25_index=None,
            )

            with (
                patch.object(main, "workspace_data_dir", return_value=data_dir),
                patch.object(
                    main,
                    "build_index",
                    side_effect=RuntimeError("staging failed"),
                ),
                patch.object(
                    main,
                    "get_workspace_index_locked",
                    return_value=workspace,
                ),
            ):
                response = client.delete("/documents/remove.pdf")

            self.assertEqual(pdf_path.read_bytes(), b"previous-content")
            self.assertEqual(list(data_dir.glob(".deleted_*.pdf")), [])

        self.assertEqual(response.status_code, 200)
        self.assertIn("已保留原索引", response.json()["error"])
        self.assertEqual(response.json()["collection"], "rag_demo_old")

    def test_delete_document_rejects_invalid_name(self):
        response = client.delete("/documents/notes.txt")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "文档名称不合法"})

    def test_upload_rebuilds_index_without_external_model(self):
        fake_page = SimpleNamespace(page_content="可读取的 PDF 文本")
        workspace = main.WorkspaceIndex(
            collection_name="rag_demo_build_test",
            vectorstore=SimpleNamespace(),
            bm25_documents=[],
            bm25_index=None,
        )

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "workspace_data_dir", return_value=Path(temp_dir)),
                patch.object(main, "PyPDFLoader") as loader,
                patch.object(main, "build_index", return_value=4) as build,
                patch.object(
                    main,
                    "get_workspace_index_locked",
                    return_value=workspace,
                ),
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
        self.assertEqual(response.json()["filename"], "knowledge.pdf")
        build.assert_called_once_with(TEST_USER.id)

    def test_upload_failure_restores_previous_pdf(self):
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pdf_path = data_dir / "knowledge.pdf"
            pdf_path.write_bytes(b"previous-pdf")
            fake_page = SimpleNamespace(page_content="可读取的 PDF 文本")
            workspace = main.WorkspaceIndex(
                collection_name="rag_demo_old",
                vectorstore=SimpleNamespace(),
                bm25_documents=[],
                bm25_index=None,
            )

            with (
                patch.object(main, "workspace_data_dir", return_value=data_dir),
                patch.object(main, "PyPDFLoader") as loader,
                patch.object(
                    main,
                    "build_index",
                    side_effect=RuntimeError("staging failed"),
                ),
                patch.object(
                    main,
                    "get_workspace_index_locked",
                    return_value=workspace,
                ),
            ):
                loader.return_value.load.return_value = [fake_page]
                response = client.post(
                    "/upload",
                    files={
                        "file": (
                            "knowledge.pdf",
                            b"%PDF-1.4-new-content",
                            "application/pdf",
                        )
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("已保留原索引", response.json()["error"])
            self.assertEqual(pdf_path.read_bytes(), b"previous-pdf")
            self.assertEqual(list(data_dir.glob(".upload_*.pdf")), [])
            self.assertEqual(list(data_dir.glob(".backup_*.pdf")), [])

    def test_upload_rejects_non_pdf(self):
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"text", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "只支持 PDF 文件"})

    def test_upload_rejects_fake_pdf_content(self):
        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "workspace_data_dir", return_value=Path(temp_dir)),
                patch.object(main, "PyPDFLoader") as loader,
            ):
                response = client.post(
                    "/upload",
                    files={
                        "file": (
                            "fake.pdf",
                            b"this is not a pdf",
                            "application/pdf",
                        )
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "文件内容不是有效的 PDF"})
        loader.assert_not_called()


class AuthAndIsolationTest(unittest.TestCase):
    def setUp(self):
        main.rate_limiter.reset()
        self.temp_dir = TemporaryDirectory()
        self.old_session_db = main.SESSION_DB
        self.old_auth_store = main.auth_store
        self.old_user_override = main.app.dependency_overrides.pop(
            main.require_user,
            None,
        )
        main.SESSION_DB = Path(self.temp_dir.name) / "sessions.sqlite3"
        main.init_session_db()
        main.auth_store = AuthStore(Path(self.temp_dir.name) / "auth.sqlite3")

    def tearDown(self):
        main.SESSION_DB = self.old_session_db
        main.auth_store = self.old_auth_store
        if self.old_user_override is not None:
            main.app.dependency_overrides[main.require_user] = self.old_user_override
        self.temp_dir.cleanup()

    def register(self, email: str) -> tuple[dict, dict]:
        response = client.post(
            "/auth/register",
            json={"email": email, "password": "password123"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        return data, {"Authorization": f"Bearer {data['access_token']}"}

    def test_protected_endpoint_requires_login(self):
        response = client.get("/documents")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "请先登录")

    def test_register_login_and_me(self):
        registered, headers = self.register("candidate@example.com")

        me = client.get("/auth/me", headers=headers)
        login = client.post(
            "/auth/login",
            json={"email": "candidate@example.com", "password": "password123"},
        )
        wrong_password = client.post(
            "/auth/login",
            json={"email": "candidate@example.com", "password": "wrong-password"},
        )

        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["user"]["email"], registered["user"]["email"])
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["user"]["id"], registered["user"]["id"])
        self.assertEqual(wrong_password.status_code, 401)

    def test_login_rate_limit_returns_429(self):
        with patch.object(main, "AUTH_RATE_LIMIT_PER_MINUTE", 1):
            first = client.post(
                "/auth/login",
                json={"email": "missing@example.com", "password": "password123"},
            )
            blocked = client.post(
                "/auth/login",
                json={"email": "missing@example.com", "password": "password123"},
            )

        self.assertEqual(first.status_code, 401)
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(
            blocked.json()["error"],
            "请求过于频繁，请稍后重试",
        )
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)

    def test_documents_and_sessions_are_user_scoped(self):
        first, first_headers = self.register("first@example.com")
        second, second_headers = self.register("second@example.com")
        first_id = first["user"]["id"]
        second_id = second["user"]["id"]
        self.assertNotEqual(main.workspace_data_dir(first_id), main.workspace_data_dir(second_id))

        user_root = Path(self.temp_dir.name) / "users"
        first_dir = user_root / first_id
        second_dir = user_root / second_id
        first_dir.mkdir(parents=True)
        second_dir.mkdir(parents=True)
        (first_dir / "first-only.pdf").write_bytes(b"first")
        (second_dir / "second-only.pdf").write_bytes(b"second")

        first_workspace = main.WorkspaceIndex(
            collection_name="rag_demo_first",
            vectorstore=SimpleNamespace(),
            bm25_documents=[SimpleNamespace(metadata={"filename": "first-only.pdf"})],
            bm25_index=None,
        )
        second_workspace = main.WorkspaceIndex(
            collection_name="rag_demo_second",
            vectorstore=SimpleNamespace(),
            bm25_documents=[SimpleNamespace(metadata={"filename": "second-only.pdf"})],
            bm25_index=None,
        )

        def data_dir(owner_id):
            return user_root / owner_id

        def workspace(owner_id):
            return first_workspace if owner_id == first_id else second_workspace

        with (
            patch.object(main, "workspace_data_dir", side_effect=data_dir),
            patch.object(main, "get_workspace_index_locked", side_effect=workspace),
        ):
            first_documents = client.get("/documents", headers=first_headers)
            second_documents = client.get("/documents", headers=second_headers)

        self.assertEqual(
            [item["filename"] for item in first_documents.json()["documents"]],
            ["first-only.pdf"],
        )
        self.assertEqual(
            [item["filename"] for item in second_documents.json()["documents"]],
            ["second-only.pdf"],
        )

        main.save_history(
            "shared-session",
            [{"role": "user", "content": "first user message"}],
            first_id,
        )
        self.assertEqual(
            main.load_history("shared-session", second_id),
            [],
        )
        with self.assertRaises(PermissionError):
            main.save_history(
                "shared-session",
                [{"role": "user", "content": "overwrite attempt"}],
                second_id,
            )


if __name__ == "__main__":
    unittest.main()
