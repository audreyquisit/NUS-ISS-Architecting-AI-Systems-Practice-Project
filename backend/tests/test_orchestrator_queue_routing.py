import unittest

from agents.orchestrator import (
    _explicit_origin_from_message,
    _normalize_queue_intent,
    _normalize_tasks,
    _queue_status_response,
    synthesizer,
)
from models.orchestration import ParsedRequest, PlannedTask


class OrchestratorQueueRoutingTests(unittest.TestCase):
    def test_explicit_origin_fallback_extracts_nearby_place(self):
        self.assertEqual(
            _explicit_origin_from_message("Hi, recommend me some food near Sengkang"),
            "Sengkang",
        )
        self.assertEqual(
            _explicit_origin_from_message("Find chicken rice near Yio Chu Kang for lunch"),
            "Yio Chu Kang",
        )

    def test_plain_queue_question_dispatches_queue_worker(self):
        parsed = ParsedRequest(intent="food_discovery")

        tasks = _normalize_tasks(
            [],
            parsed,
            "How are the queues like?",
        )

        self.assertIn("queue", [task.agent for task in tasks])

    def test_queue_intent_normalizes_and_dispatches_queue_worker(self):
        parsed = ParsedRequest(intent="queue")
        normalized, queue_requested = _normalize_queue_intent(parsed)

        tasks = _normalize_tasks(
            [],
            normalized,
            "Find somewhere with a short queue",
            queue_requested=queue_requested,
        )

        self.assertEqual(normalized.intent, "food_discovery")
        self.assertTrue(queue_requested)
        self.assertIn("queue", [task.agent for task in tasks])

    def test_explicit_queue_task_is_preserved(self):
        parsed = ParsedRequest(intent="food_discovery")
        normalized, queue_requested = _normalize_queue_intent(parsed)
        planned_queue = PlannedTask(agent="queue", task="Compare mock queues")

        tasks = _normalize_tasks(
            [planned_queue],
            normalized,
            "Find a stall with a manageable queue",
            queue_requested=queue_requested,
        )

        self.assertEqual(
            [task for task in tasks if task.agent == "queue"], [planned_queue]
        )

    def test_queue_question_returns_verified_queue_summary(self):
        reply = _queue_status_response(
            "How are the queues like?",
            [
                {
                    "agent": "location",
                    "payload": {
                        "selected_centres": [
                            {"centre_id": "newton_food_centre"},
                        ],
                    },
                },
                {
                    "agent": "queue",
                    "verified": True,
                    "candidates": [
                        {
                            "hawker_centre_id": "newton_food_centre",
                            "hawker_centre": "Newton Food Centre",
                            "queue_minutes": 12,
                            "crowd_level": "Moderate",
                        },
                        {
                            "hawker_centre_id": "newton_food_centre",
                            "hawker_centre": "Newton Food Centre",
                            "queue_minutes": 18,
                            "crowd_level": "High",
                        },
                        {
                            "hawker_centre_id": "another_centre",
                            "hawker_centre": "Other Centre",
                            "queue_minutes": 3,
                            "crowd_level": "Low",
                        },
                    ],
                },
            ],
        )

        self.assertIn("SIMULATED QUEUE DATA", reply)
        self.assertIn("1. Newton Food Centre", reply)
        self.assertIn(
            "   - Average estimated wait: about 15 minutes across 2 stalls.",
            reply,
        )
        self.assertIn("   - Most common crowd level: Moderate.", reply)
        self.assertIn("Newton Food Centre", reply)
        self.assertNotIn("Other Centre", reply)

    def test_named_stall_queue_question_returns_individual_stall_stats(self):
        reply = _queue_status_response(
            "I want to know the queue for Ang Mo Kio Ave 10 Blk 453A, wanton mee stall",
            [
                {
                    "agent": "location",
                    "payload": {
                        "selected_centres": [
                            {"centre_id": "ang_mo_kio_ave_10_blk_453a"},
                        ],
                    },
                },
                {
                    "agent": "queue",
                    "verified": True,
                    "candidates": [
                        {
                            "hawker_centre_id": "ang_mo_kio_ave_10_blk_453a",
                            "hawker_centre": "Ang Mo Kio Ave 10 Blk 453A (Chong Boon Market and Food Centre)",
                            "stall_name": "Demo Wanton mee Stall",
                            "queue_minutes": 7,
                            "crowd_level": "Moderate",
                        },
                        {
                            "hawker_centre_id": "ang_mo_kio_ave_10_blk_453a",
                            "hawker_centre": "Ang Mo Kio Ave 10 Blk 453A (Chong Boon Market and Food Centre)",
                            "stall_name": "Demo Chicken rice Stall",
                            "queue_minutes": 19,
                            "crowd_level": "High",
                        },
                    ],
                },
            ],
        )

        self.assertIn("Demo Wanton mee Stall", reply)
        self.assertIn("about 7 minutes estimated wait", reply)
        self.assertIn("Crowd level: Moderate", reply)
        self.assertNotIn("average", reply.casefold())
        self.assertNotIn("Demo Chicken rice Stall", reply)

    def test_queue_question_without_verified_data_does_not_claim_live_status(self):
        reply = _queue_status_response(
            "How are the queues like?",
            [{"agent": "queue", "verified": True, "candidates": []}],
        )

        self.assertIn("couldn't find queue estimates", reply)
        self.assertIn("not a live service", reply)

    def test_synthesizer_prefers_queue_data_over_empty_menu_fallback(self):
        result = synthesizer({
            "user_request": "How are the queues like near Yio Chu Kang?",
            "parsed_request": {"intent": "food_discovery"},
            "verified_results": [
                {
                    "agent": "location",
                    "payload": {
                        "selected_centres": [
                            {"centre_id": "newton_food_centre"},
                        ],
                    },
                },
                {
                    "agent": "candidate_join",
                    "verified": True,
                    "candidates": [],
                },
                {
                    "agent": "queue",
                    "verified": True,
                    "candidates": [{
                        "hawker_centre_id": "newton_food_centre",
                        "hawker_centre": "Newton Food Centre",
                        "queue_minutes": 12,
                        "crowd_level": "Moderate",
                    }],
                },
            ],
        })

        self.assertIn("Newton Food Centre", result["final_recommendation"])
        self.assertNotIn("don't have menu listings", result["final_recommendation"])


if __name__ == "__main__":
    unittest.main()