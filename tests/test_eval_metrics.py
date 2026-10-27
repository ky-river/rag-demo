import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.metrics import (
    format_ratio,
    latency_summary,
    percentile,
    render_markdown,
    write_reports,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_percentile_interpolates_sorted_values(self):
        values = [400, 100, 300, 200]

        self.assertEqual(percentile(values, 0), 100)
        self.assertEqual(percentile(values, 50), 250)
        self.assertEqual(percentile(values, 100), 400)
        self.assertEqual(percentile([], 95), 0)

    def test_latency_summary_reports_tail_latency(self):
        summary = latency_summary([100, 200, 300, 400])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["average_ms"], 250)
        self.assertEqual(summary["p50_ms"], 250)
        self.assertEqual(summary["p95_ms"], 385)
        self.assertEqual(summary["max_ms"], 400)

    def test_reports_include_failures_in_json_and_markdown(self):
        report = {
            "title": "RAG 检索评测报告",
            "status": "failed",
            "generated_at": "2026-10-27T00:00:00+00:00",
            "meta": {"套件": "smoke"},
            "metrics": [{"name": "Hit@4", "value": "9/10 (90.0%)"}],
            "categories": [
                {
                    "name": "project",
                    "metrics": [{"name": "MRR", "value": "0.900"}],
                }
            ],
            "failures": [
                {
                    "question": "测试问题",
                    "reason": "未命中预期来源",
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_reports(
                report,
                temp_dir,
                "retrieval-smoke",
            )
            saved = json.loads(Path(json_path).read_text(encoding="utf-8"))
            markdown = Path(markdown_path).read_text(encoding="utf-8")

        self.assertEqual(saved["failures"][0]["question"], "测试问题")
        self.assertIn("状态：未通过", markdown)
        self.assertIn("Hit@4 | 9/10", markdown)
        self.assertIn("测试问题：未命中预期来源", markdown)

    def test_ratio_and_markdown_handle_empty_results(self):
        markdown = render_markdown(
            {
                "title": "空评测",
                "status": "passed",
                "generated_at": "2026-10-27T00:00:00+00:00",
                "meta": {},
                "metrics": [],
                "failures": [],
            }
        )

        self.assertEqual(format_ratio(0, 0), "0/0 (0.0%)")
        self.assertIn("## 失败明细\n\n无。", markdown)


if __name__ == "__main__":
    unittest.main()
