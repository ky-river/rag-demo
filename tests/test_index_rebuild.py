import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app.main as main


def fake_vectorstore(count: int):
    return SimpleNamespace(
        add_documents=Mock(),
        delete_collection=Mock(),
        _collection=SimpleNamespace(count=lambda: count),
    )


class AtomicIndexRebuildTest(unittest.TestCase):
    def run_build(self, staging_store):
        page = SimpleNamespace(
            page_content="索引文本",
            metadata={"filename": "project.pdf"},
        )
        chunk = SimpleNamespace(
            page_content="索引文本",
            metadata={"filename": "project.pdf", "page": 0},
        )
        old_store = fake_vectorstore(1)
        new_bm25 = object()

        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "project.pdf").write_bytes(b"%PDF-1.4")
            (data_dir / ".backup_project.pdf").write_bytes(b"backup")
            with (
                patch.object(main, "DATA_DIR", data_dir),
                patch.object(main, "PyPDFLoader") as loader,
                patch.object(
                    main.splitter,
                    "split_documents",
                    return_value=[chunk],
                ),
                patch.object(main, "Chroma", return_value=staging_store),
                patch.object(main, "build_bm25_index", return_value=new_bm25),
                patch.object(
                    main,
                    "persist_active_collection_name",
                ) as persist,
                patch.object(main, "vectorstore", old_store),
                patch.object(main, "active_collection_name", "rag_demo"),
                patch.object(main, "bm25_documents", []),
                patch.object(main, "bm25_index", None),
            ):
                loader.return_value.load.return_value = [page]
                error = None
                result = None
                try:
                    result = main.build_index()
                except Exception as exc:
                    error = exc
                current_vectorstore = main.vectorstore
                current_collection_name = main.active_collection_name

        return {
            "result": result,
            "error": error,
            "old_store": old_store,
            "persist": persist,
            "new_bm25": new_bm25,
            "loader": loader,
            "current_vectorstore": current_vectorstore,
            "current_collection_name": current_collection_name,
        }

    def test_success_switches_collection_after_validation(self):
        staging_store = fake_vectorstore(1)
        result = self.run_build(staging_store)

        self.assertEqual(result["result"], 1)
        result["loader"].assert_called_once()
        staging_store.add_documents.assert_called_once()
        staging_store.delete_collection.assert_not_called()
        result["old_store"].delete_collection.assert_called_once_with()
        result["persist"].assert_called_once()
        collection_name = result["persist"].call_args.args[0]
        self.assertTrue(collection_name.startswith("rag_demo_build_"))
        self.assertIs(result["current_vectorstore"], staging_store)
        self.assertEqual(result["current_collection_name"], collection_name)

    def test_validation_failure_keeps_old_index(self):
        staging_store = fake_vectorstore(0)
        result = self.run_build(staging_store)

        self.assertIsInstance(result["error"], RuntimeError)
        self.assertIn("索引校验失败", str(result["error"]))
        staging_store.add_documents.assert_called_once()
        staging_store.delete_collection.assert_called_once_with()
        result["persist"].assert_not_called()
        self.assertIs(result["current_vectorstore"], result["old_store"])
        self.assertEqual(result["current_collection_name"], "rag_demo")


if __name__ == "__main__":
    unittest.main()
