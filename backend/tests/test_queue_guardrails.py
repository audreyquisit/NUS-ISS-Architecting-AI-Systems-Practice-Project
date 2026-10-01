import unittest
from unittest.mock import patch

from agents.verifier import _verify_worker_result


def _candidate(
    stall_id="stall-a",
    centre_id="centre-a",
    queue_minutes=8,
    crowd_level="Moderate",
    source="synthetic_queue_mock",
    queue_is_mock=True,
):
    return {
        "stall_id": stall_id,
        "stall_name": "Untrusted stall label",
        "hawker_centre_id": centre_id,
        "hawker_centre": "Untrusted centre label",
        "queue_minutes": queue_minutes,
        "crowd_level": crowd_level,
        "queue_score": 72,
        "queue_is_mock": queue_is_mock,
        "queue_mock_notice": "Simulated queue estimate; not a live queue.",
        "source": source,
    }


class QueueGuardrailTests(unittest.TestCase):
    @patch("agents.verifier.get_menu_stall_record")
    def test_verifier_rejects_bad_provenance_identity_crowd_and_hard_limit(
        self, get_stall
    ):
        source_records = {
            "stall-a": {
                "stall_id": "stall-a",
                "stall_name": "Verified Stall",
                "hawker_centre_id": "centre-a",
                "hawker_centre": "Verified Centre",
            },
            "stall-b": {
                "stall_id": "stall-b",
                "stall_name": "Other Centre Stall",
                "hawker_centre_id": "centre-b",
                "hawker_centre": "Other Centre",
            },
            "stall-c": {
                "stall_id": "stall-c",
                "stall_name": "Wrong Crowd Stall",
                "hawker_centre_id": "centre-a",
                "hawker_centre": "Verified Centre",
            },
            "stall-d": {
                "stall_id": "stall-d",
                "stall_name": "Long Queue Stall",
                "hawker_centre_id": "centre-a",
                "hawker_centre": "Verified Centre",
            },
        }
        get_stall.side_effect = lambda stall_id: source_records.get(stall_id)
        result = {
            "agent": "queue",
            "confidence": 0.99,
            "candidates": [
                _candidate(),
                _candidate(stall_id="stall-b"),
                _candidate(stall_id="stall-c", queue_minutes=18, crowd_level="Low"),
                _candidate(stall_id="stall-d", queue_minutes=12),
                _candidate(stall_id="unknown"),
                _candidate(stall_id="stall-a", source="live_queue_feed"),
            ],
        }
        result["candidates"][1]["hawker_centre_id"] = "centre-a"
        result["candidates"][3]["queue_is_mock"] = False

        checked = _verify_worker_result(result, {"max_queue_min": 10})

        self.assertTrue(checked["verified"])
        self.assertEqual(len(checked["candidates"]), 1)
        candidate = checked["candidates"][0]
        self.assertEqual(candidate["stall_name"], "Verified Stall")
        self.assertEqual(candidate["hawker_centre"], "Verified Centre")
        self.assertEqual(checked["confidence"], 0.17)
        self.assertEqual(checked["removed_candidate_count"], 5)
        self.assertIn("not prediction accuracy", checked["payload"]["confidence_basis"])

    @patch("agents.verifier.get_menu_stall_record")
    def test_invalid_hard_limit_fails_closed(self, get_stall):
        get_stall.return_value = {
            "stall_id": "stall-a",
            "stall_name": "Verified Stall",
            "hawker_centre_id": "centre-a",
            "hawker_centre": "Verified Centre",
        }
        result = {"agent": "queue", "candidates": [_candidate()]}

        checked = _verify_worker_result(result, {"max_queue_min": -1})

        self.assertFalse(checked["verified"])
        self.assertEqual(checked["status"], "unavailable")
        self.assertEqual(checked["candidates"], [])


if __name__ == "__main__":
    unittest.main()