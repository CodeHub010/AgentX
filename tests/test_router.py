"""Tests for deterministic AgentX task routing."""

import unittest

from agentx.core.router import route_task


class TaskRouterTests(unittest.TestCase):
    def test_valid_task_types_map_to_expected_routes(self):
        expected = {
            "web_research": "research",
            "document_qa": "knowledge",
            "combined_research": "combined",
            "unknown": "clarification",
        }
        for task_type, route in expected.items():
            with self.subTest(task_type=task_type):
                self.assertEqual(route_task(task_type), route)

    def test_invalid_or_missing_task_type_uses_clarification(self):
        for task_type in ("other", None, 42, [], {}):
            with self.subTest(task_type=task_type):
                self.assertEqual(route_task(task_type), "clarification")


if __name__ == "__main__":
    unittest.main()
