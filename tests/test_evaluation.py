import unittest

from app.evaluation import load_cases, run_retrieval_evaluation


class RetrievalEvaluationTest(unittest.TestCase):
    def test_extended_suite_passes_with_expected_sources(self):
        cases = load_cases("extended")

        def search(question):
            case = next(item for item in cases if item["question"] == question)
            if case.get("expect_no_sources"):
                return []
            return [
                {
                    "filename": expected["filename"],
                    "content": " ".join(expected["any_keywords"]),
                }
                for expected in case["expected_sources"]
            ]

        report = run_retrieval_evaluation("extended", search)

        self.assertEqual(report["status"], "passed")
        self.assertIn("Hit@4", {item["name"] for item in report["metrics"]})
        self.assertEqual(report["failures"], [])
        self.assertEqual(len(report["cases"]), len(cases))
        self.assertTrue(any(item["name"] == "P95 检索延迟" for item in report["metrics"]))

    def test_retrieval_errors_are_reported_as_failures(self):
        def search(_question):
            raise RuntimeError("index unavailable")

        report = run_retrieval_evaluation("smoke", search)

        self.assertEqual(report["status"], "failed")
        self.assertGreater(len(report["failures"]), 0)
        self.assertIn("index unavailable", report["failures"][0]["reason"])

    def test_unknown_suite_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的评测套件"):
            load_cases("missing")


if __name__ == "__main__":
    unittest.main()
