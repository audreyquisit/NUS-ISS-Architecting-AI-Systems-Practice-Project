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
    @patch("agents.queue_agent._agentic_queue_plan")
    def test_invalid_queue_limits_fail_closed_before_planning(self, planner):
        for invalid_limit in (-1, 4.5, "10", True):
            with self.subTest(queue_limit=invalid_limit):
                result = run(
                    "Find a short queue",
                    parsed_request={"max_queue_min": invalid_limit},
                )
                self.assertEqual(result["candidates"], [])
                self.assertEqual(result["confidence"], 0.0)
                self.assertTrue(any(
                    "non-negative whole number" in item
                    for item in result["limitations"]
                ))
        planner.assert_not_called()

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

    @patch("agents.queue_agent._rank_candidates", side_effect=RuntimeError("offline"))
    @patch("agents.queue_agent._agentic_queue_plan")
    def test_google_places_result_keeps_proxy_provenance(self, agentic_plan, _rank):
        place = {
            "_agent_candidate_id": "queue-0",
            "stall_id": "places/chinatown-complex",
            "stall_name": "Chinatown Complex",
            "hawker_centre": "Chinatown Complex",
            "hawker_centre_id": "smith_street_blk_335_chinatown_complex_market",
            "queue_minutes": 3,
            "crowd_level": "Low",
            "queue_score": 91,
            "source": "google_places",
            "data_origin": "google_places",
            "place_id": "places/chinatown-complex",
            "queue_is_mock": False,
            "queue_estimate_type": "popularity_proxy",
        }
        agentic_plan.return_value = ([place], [])

        result = run("Check queues for Chinatown Complex")

        self.assertEqual(result["candidates"][0]["source"], "google_places")
        self.assertEqual(result["candidates"][0]["stall_id"], "places/chinatown-complex")
        self.assertEqual(result["evidence"][0]["source"], "google_places")
        self.assertTrue(any(
            "not observed queues" in item for item in result["limitations"]
        ))

    @patch("services.google_places_queue_feed.requests.post")
    @patch.dict(
        "os.environ",
        {
            "QUEUE_FEED_PROVIDER": "google_places",
            "GOOGLE_PLACES_API_KEY": "test-key",
        },
        clear=False,
    )
    def test_google_places_queue_provider_maps_place_results(self, mock_post):
        from services.queue_provider import get_live_queue_feed

        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {
            "places": [
                {
                    "id": "places/maxwell-food-centre",
                    "displayName": {"text": "Maxwell Food Centre"},
                    "formattedAddress": "1 Kadayanallur St, Singapore 069184",
                    "rating": 4.7,
                    "userRatingCount": 150,
                    "businessStatus": "OPERATIONAL",
                }
            ],
        }

        records = get_live_queue_feed("Maxwell Food Centre")

        self.assertEqual(records[0]["source"], "google_places")
        self.assertEqual(records[0]["stall_name"], "Maxwell Food Centre")
        self.assertIn("queue_minutes", records[0])
        self.assertIn("crowd_level", records[0])
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(
            args[0],
            "https://places.googleapis.com/v1/places:searchText",
        )
        self.assertEqual(kwargs["headers"]["X-Goog-Api-Key"], "test-key")
        self.assertIn("places.displayName", kwargs["headers"]["X-Goog-FieldMask"])
        self.assertEqual(kwargs["json"], {"textQuery": "Maxwell Food Centre"})


if __name__ == "__main__":
    unittest.main()