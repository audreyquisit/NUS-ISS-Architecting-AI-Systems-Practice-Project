import unittest
from unittest.mock import patch

from agents.queue_agent import (
    QueueAssessment,
    QueueSelection,
    _agentic_queue_plan,
    run,
)


def _feed():
    return [
        {
            "stall_name": "Noodle Stall",
            "hawker_centre": "Maxwell Food Centre",
            "queue_minutes": 4,
            "crowd_level": "Low",
            "queue_score": 86,
        },
        {
            "stall_name": "Rice Stall",
            "hawker_centre": "Maxwell Food Centre",
            "queue_minutes": 8,
            "crowd_level": "Moderate",
            "queue_score": 72,
        },
        {
            "stall_name": "Popular Stall",
            "hawker_centre": "Maxwell Food Centre",
            "queue_minutes": 16,
            "crowd_level": "High",
            "queue_score": 51,
        },
    ]


class QueueAgentTests(unittest.TestCase):
    @patch("agents.queue_agent.get_menu_stall_id", return_value="stall-id")
    @patch("agents.queue_agent.get_live_queue_feed", side_effect=lambda task: _feed())
    @patch("agents.queue_agent._llm")
    def test_planner_chooses_lookup_estimate_and_limit_tools(
        self, llm_factory, _get_feed, _get_stall_id
    ):
        class ScriptedModel:
            def __init__(self):
                self.step = 0

            def bind_tools(self, _tools):
                return self

            def invoke(self, _messages):
                from types import SimpleNamespace

                self.step += 1
                tool_calls = {
                    1: [{
                        "name": "lookup_queue_records",
                        "args": {"centre_names": ["Maxwell Food Centre"]},
                        "id": "lookup-1",
                    }],
                    2: [{
                        "name": "estimate_queue_scenario",
                        "args": {"candidate_ids": ["queue-0", "queue-1"]},
                        "id": "estimate-1",
                    }],
                    3: [{
                        "name": "apply_queue_limit",
                        "args": {"candidate_ids": ["queue-0", "queue-1"]},
                        "id": "limit-1",
                    }],
                }.get(self.step, [])
                return SimpleNamespace(tool_calls=tool_calls)

        llm_factory.return_value = ScriptedModel()
        candidates, trace = _agentic_queue_plan(
            "short queue for lunch on Saturday in rain",
            "Find a short queue",
            {"max_queue_min": 10, "time_period": "lunch"},
            ["Maxwell Food Centre"],
            ["maxwell-id"],
        )

        self.assertEqual(
            [item["queue_minutes"] for item in candidates], [9]
        )
        self.assertTrue(any(
            item.get("tool") == "estimate_queue_scenario" for item in trace
        ))
        self.assertTrue(any(
            item.get("tool") == "apply_queue_limit" for item in trace
        ))
        self.assertTrue(all(
            item.get("source") == "synthetic_queue_mock" for item in candidates
        ))

    @patch("agents.queue_agent.get_menu_stall_id", return_value="stall-id")
    @patch("agents.queue_agent.get_live_queue_feed", side_effect=lambda task: _feed())
    @patch("agents.queue_agent._rank_candidates")
    @patch("agents.queue_agent._agentic_queue_plan")
    def test_filters_queue_limit_and_ignores_unknown_model_ids(
        self, _agentic_plan, rank_candidates, _get_feed, _get_stall_id
    ):
        candidates = _feed()[:2]
        for index, item in enumerate(candidates):
            item["_agent_candidate_id"] = f"queue-{index}"
            item["hawker_centre_id"] = "maxwell-id"
            item["stall_id"] = "stall-id"
            item["source"] = "mock_queue_feed"
        _agentic_plan.return_value = (candidates, [])
        rank_candidates.return_value = QueueAssessment(
            ranked_candidates=[
                QueueSelection(
                    candidate_id="queue-1", rank=1, rationale="Eight-minute wait."
                ),
                QueueSelection(
                    candidate_id="invented-id", rank=2, rationale="Invalid."
                ),
            ],
            reasoning="The eight-minute option best fits the request.",
            confidence=0.8,
        )

        result = run(
            "Find a short queue",
            parsed_request={"max_queue_min": 10},
            candidate_centres=["Maxwell Food Centre"],
            candidate_centre_ids=["maxwell-id"],
        )

        passed_candidates = rank_candidates.call_args.args[3]
        self.assertEqual(
            [item["queue_minutes"] for item in passed_candidates], [4, 8]
        )
        self.assertEqual(
            [item["queue_minutes"] for item in result["candidates"]], [8, 4]
        )
        self.assertEqual(
            [item["queue_rank"] for item in result["candidates"]], [1, 2]
        )
        self.assertNotIn(
            "invented-id",
            [item.get("_agent_candidate_id") for item in result["candidates"]],
        )
        self.assertTrue(
            any("mock lookup" in item for item in result["limitations"])
        )

    @patch("agents.queue_agent.get_menu_stall_id", return_value="stall-id")
    @patch("agents.queue_agent.get_live_queue_feed", side_effect=lambda task: _feed())
    @patch("agents.queue_agent._rank_candidates", side_effect=RuntimeError("offline"))
    @patch("agents.queue_agent._agentic_queue_plan", side_effect=RuntimeError("offline"))
    def test_model_failure_returns_deterministic_grounded_fallback(
        self, _agentic_plan, _rank_candidates, _get_feed, _get_stall_id
    ):
        result = run("Find a short queue")

        self.assertEqual(
            [item["queue_minutes"] for item in result["candidates"]], [4, 8, 16]
        )
        self.assertEqual(
            [item["queue_rank"] for item in result["candidates"]], [1, 2, 3]
        )
        self.assertTrue(
            any("assessment was unavailable" in item for item in result["limitations"])
        )
        self.assertIn("deterministically", result["reasoning"])


if __name__ == "__main__":
    unittest.main()