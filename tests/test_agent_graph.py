import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app.main as main


class AgentToolSelectionTest(unittest.TestCase):
    def test_model_selection_keeps_arguments_and_routing(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="knowledge_search",
                arguments='{"question":"项目技术栈"}',
            )
        )
        message = SimpleNamespace(tool_calls=[tool_call], content=None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        with patch.object(main.client.chat.completions, "create", return_value=response):
            result = main.select_agent_tool("项目技术栈", [])

        self.assertEqual(
            result,
            ("knowledge_search", {"question": "项目技术栈"}, "model"),
        )

    def test_no_tool_call_uses_fallback(self):
        message = SimpleNamespace(tool_calls=None, content="请补充信息")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        with patch.object(main.client.chat.completions, "create", return_value=response):
            result = main.select_agent_tool("这个项目使用了哪些技术？", [])

        self.assertEqual(result[0], "knowledge_search")
        self.assertEqual(result[2], "fallback")


class AgentGraphTest(unittest.TestCase):
    def test_all_tool_edges_reach_matching_node(self):
        cases = [
            ("knowledge_search", {"question": "项目技术栈"}, "knowledge"),
            ("create_study_plan", {"topic": "LangGraph"}, "study"),
            ("resume_match_guidance", {}, "resume"),
        ]

        for tool, arguments, expected_answer in cases:
            with self.subTest(tool=tool):
                with (
                    patch.object(
                        main,
                        "select_agent_tool",
                        return_value=(tool, arguments, "model"),
                    ),
                    patch.object(
                        main,
                        "execute_agent_tool",
                        return_value={"answer": expected_answer, "sources": []},
                    ) as execute,
                ):
                    state = main.agent_graph.invoke(
                        {"message": "test", "history": []}
                    )

                self.assertEqual(state["tool"], tool)
                self.assertEqual(state["routing"], "model")
                self.assertEqual(state["answer"], expected_answer)
                execute.assert_called_once_with(tool, arguments, [])


class KnowledgeRetrievalTest(unittest.TestCase):
    def test_cross_encoder_reranks_fused_candidates(self):
        first = {
            "document": SimpleNamespace(
                metadata={"filename": "project.pdf", "page": 0},
                page_content="弱相关片段",
            ),
            "vector_score": 0.8,
            "lexical_score": 1.0,
            "hybrid_score": 0.5,
            "retrieval_method": "vector+bm25",
        }
        second = {
            "document": SimpleNamespace(
                metadata={"filename": "project.pdf", "page": 1},
                page_content="直接回答问题的片段",
            ),
            "vector_score": 0.5,
            "lexical_score": 2.0,
            "hybrid_score": 0.4,
            "retrieval_method": "vector+bm25",
        }
        fake_reranker = SimpleNamespace(
            rerank=lambda question, documents, batch_size: [0.1, 0.9]
        )

        with patch.object(main, "RERANK_CANDIDATES", 2):
            reranked = main.rerank_items(
                "问题",
                [first, second],
                reranker_override=fake_reranker,
            )

        self.assertEqual(reranked[0]["document"].page_content, "直接回答问题的片段")
        self.assertEqual(reranked[0]["rerank_score"], 0.9)
        self.assertEqual(reranked[0]["pre_rerank_rank"], 2)

    def test_reranker_failure_keeps_rrf_order(self):
        items = [
            {
                "document": SimpleNamespace(
                    metadata={"filename": "project.pdf", "page": 0},
                    page_content="片段 A",
                ),
                "hybrid_score": 0.5,
            },
            {
                "document": SimpleNamespace(
                    metadata={"filename": "project.pdf", "page": 1},
                    page_content="片段 B",
                ),
                "hybrid_score": 0.4,
            },
        ]

        def fail_rerank(question, documents, batch_size):
            raise RuntimeError("model unavailable")

        fake_reranker = SimpleNamespace(rerank=fail_rerank)
        reranked = main.rerank_items(
            "问题",
            items,
            reranker_override=fake_reranker,
        )

        self.assertEqual(reranked, items)

    def test_top_k_and_relevance_threshold_are_applied(self):
        documents = [
            (
                SimpleNamespace(
                    metadata={"filename": "project.pdf", "page": 0},
                    page_content="向量数据库使用 Chroma",
                ),
                0.19,
            ),
            (
                SimpleNamespace(
                    metadata={"filename": "project.pdf", "page": 1},
                    page_content="不相关片段",
                ),
                1.1,
            ),
        ]
        model_message = SimpleNamespace(content="使用 Chroma")
        model_response = SimpleNamespace(
            choices=[SimpleNamespace(message=model_message)]
        )

        with (
            patch.object(main, "bm25_index", None),
            patch.object(
                main.vectorstore,
                "similarity_search_with_score",
                return_value=documents,
            ) as search,
            patch.object(
                main.client.chat.completions,
                "create",
                return_value=model_response,
            ),
        ):
            result = main.answer_from_knowledge_base("向量数据库是什么？")

        search.assert_called_once_with("向量数据库是什么？", k=12)
        self.assertEqual(result["answer"], "使用 Chroma")
        self.assertEqual(len(result["sources"]), 1)
        self.assertAlmostEqual(result["sources"][0]["relevance_score"], 0.8656)

    def test_no_relevant_document_skips_model_call(self):
        documents = [
            (
                SimpleNamespace(
                    metadata={"filename": "project.pdf", "page": 0},
                    page_content="不相关片段",
                ),
                1.8,
            )
        ]

        with (
            patch.object(main, "bm25_index", None),
            patch.object(
                main.vectorstore,
                "similarity_search_with_score",
                return_value=documents,
            ),
            patch.object(main.client.chat.completions, "create") as model_call,
        ):
            result = main.answer_from_knowledge_base("公司年收入是多少？")

        self.assertEqual(result, {"answer": "当前资料中没有找到答案", "sources": []})
        model_call.assert_not_called()

    def test_bm25_only_document_enters_hybrid_results(self):
        vector_document = SimpleNamespace(
            metadata={"filename": "project.pdf", "page": 0},
            page_content="向量检索命中片段",
        )
        lexical_document = SimpleNamespace(
            metadata={"filename": "project.pdf", "page": 1},
            page_content="SQLite 会话存储",
        )
        fake_index = SimpleNamespace(get_scores=lambda tokens: [0.0, 5.0])

        with (
            patch.object(main, "bm25_index", fake_index),
            patch.object(main, "bm25_documents", [vector_document, lexical_document]),
            patch.object(
                main.vectorstore,
                "similarity_search_with_score",
                return_value=[(vector_document, 0.1)],
            ),
        ):
            _, sources = main.retrieve_knowledge("多轮会话保存在哪里？")

        methods = {source["retrieval_method"] for source in sources}
        self.assertEqual(methods, {"vector", "bm25"})
        self.assertIsNone(
            next(
                source["relevance_score"]
                for source in sources
                if source["retrieval_method"] == "bm25"
            )
        )


if __name__ == "__main__":
    unittest.main()
