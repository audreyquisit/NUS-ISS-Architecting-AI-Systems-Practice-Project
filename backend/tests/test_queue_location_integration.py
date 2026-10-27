"""Offline checks at the Location/Google Queue integration boundaries."""

import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import orchestrator, verifier
from models.orchestration import PlannedTask
from services import google_places_queue_feed as google_feed


def google_candidate(centre):
    return {
        "stall_id": "places/" + centre,
        "stall_name": "Fixture " + centre,
        "hawker_centre_id": centre,
        "hawker_centre": "Fixture " + centre,
        "queue_minutes": 3,
        "crowd_level": "Low",
        "queue_score": 91,
        "source": "google_places",
        "data_origin": "google_places",
        "place_id": "places/" + centre,
        "queue_is_mock": False,
        "queue_estimate_type": "popularity_proxy",
        "record_type": "place",
        "business_status": "OPERATIONAL",
    }


class QueueLocationIntegrationTests(unittest.TestCase):
    @staticmethod
    def results(selected, candidates):
        return [
            {"agent": "location", "payload": {
                "selected_centres": [{"centre_id": key} for key in selected],
            }},
            {"agent": "queue", "verified": True, "candidates": candidates,
             "evidence": [{"data_sources": ["google_places"]}]},
        ]

    def test_synthesis_rejects_google_rows_outside_empty_or_partial_location_scope(self):
        for selected in ([], ["first"]):
            with self.subTest(selected=selected):
                reply = orchestrator._queue_status_response("How are the queues?",
                    self.results(selected, [google_candidate("rejected")]))
                self.assertNotIn("Fixture rejected", reply or "")
                self.assertNotIn("about 3 minutes", reply or "")
                self.assertNotIn("simulated demo data", reply or "")

    def test_google_synthesis_preserves_location_order_and_excludes_other_centres(self):
        reply = orchestrator._queue_status_response("How are the queues?", self.results(
            ["first", "second"], [google_candidate(key) for key in ("second", "rejected", "first")]))
        self.assertLess(reply.index("Fixture first"), reply.index("Fixture second"))
        self.assertNotIn("Fixture rejected", reply)
        self.assertIn("not observed queue times", reply)
        self.assertIn("not stall-specific waits", reply)

    def test_verifier_distinguishes_empty_approval_from_absent_scope(self):
        centres = [{"id": key, "name": "Fixture " + key} for key in ("first", "rejected")]
        rows = [google_candidate(key) for key in ("first", "rejected")]
        with patch.object(verifier, "load_hawker_centres", return_value=centres):
            for scope, expected in ((None, ["first", "rejected"]), (set(), []), ({"first"}, ["first"])):
                with self.subTest(scope=scope):
                    checked = verifier._verify_worker_result(
                        {"agent": "queue", "candidates": rows}, {}, approved_centres=scope)
                    self.assertEqual([row["hawker_centre_id"] for row in checked["candidates"]], expected)
                    self.assertIn("applicable Location approval", checked["verification"])

    def test_place_proxy_cannot_stand_in_for_verified_food_stall(self):
        candidate = google_candidate("first")
        candidate["stall_id"] = "first:stall"
        results = self.results(["first"], [candidate]) + [{
            "agent": "candidate_join", "verified": True,
            "candidates": [{"stall_id": "first:stall", "menu_item_id": "first:rice"}],
        }]
        self.assertIsNone(orchestrator._queue_status_response(
            "Find chicken rice and check queues", results, {"menu_item_preferences": ["chicken rice"]}))

    def test_place_name_is_not_verified_stall_identity(self):
        reply = orchestrator._queue_status_response("How are the queues for this stall?",
            self.results(["first"], [google_candidate("first")]),
            {"stall_name_preferences": ["Fixture first"]})
        self.assertNotIn("about 3 minutes", reply or "")
        self.assertIn("stall-specific", reply or "")

    def test_mixed_providers_keep_their_distinct_provenance(self):
        simulated = {**google_candidate("second"), "source": "synthetic_queue_mock",
                     "stall_name": "Fixture simulated stall", "queue_is_mock": True}
        reply = orchestrator._queue_status_response("How are the queues?",
            self.results(["second", "first"], [google_candidate("first"), simulated]))
        self.assertIn("SIMULATED QUEUE DATA", reply)
        self.assertIn("Popularity-based proxy", reply)
        self.assertLess(reply.index("Fixture second"), reply.index("Fixture first"))

    def test_closed_missing_and_unknown_google_status_cannot_verify_as_short_wait(self):
        centre = {"id": "first", "name": "Fixture first"}
        with patch.object(verifier, "load_hawker_centres", return_value=[centre]):
            for status in (None, "CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY", "UNKNOWN"):
                with self.subTest(status=status):
                    candidate = google_candidate("first")
                    candidate["business_status"] = status
                    checked = verifier._verify_worker_result({"agent": "queue", "candidates": [candidate]}, {})
                    self.assertEqual(checked["candidates"], [])
            candidate = {**google_candidate("first"), "queue_minutes": 0, "queue_score": 100}
            checked = verifier._verify_worker_result({"agent": "queue", "candidates": [candidate]}, {})
            self.assertEqual(checked["candidates"], [])

    def test_google_adapter_omits_unavailable_places_and_retains_operational_status(self):
        place = {"id": "places/first", "displayName": {"text": "Fixture first"},
                 "rating": 4.0, "userRatingCount": 100}
        with patch.dict("os.environ", {"GOOGLE_PLACES_API_KEY": "offline-test-key"}), \
             patch.object(google_feed.requests, "post") as post:
            post.return_value.json.return_value = {"places": [
                place, {**place, "businessStatus": "CLOSED_TEMPORARILY"},
                {**place, "businessStatus": "OPERATIONAL"},
            ]}
            rows = google_feed.get_live_queue_feed("Fixture first")
        post.assert_called_once()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["business_status"], "OPERATIONAL")
        self.assertGreater(rows[0]["queue_minutes"], 0)
        self.assertEqual(rows[0]["queue_estimate_type"], "popularity_proxy")

    def test_malformed_google_centre_identity_is_rejected_without_exception(self):
        with patch.object(verifier, "load_hawker_centres", return_value=[{"id": "first", "name": "Fixture first"}]):
            for invalid in ([], {}, True, None, 123):
                with self.subTest(identity=invalid):
                    candidate = {**google_candidate("first"), "hawker_centre_id": invalid}
                    checked = verifier._verify_worker_result({"agent": "queue", "candidates": [candidate]}, {})
                    self.assertEqual(checked["candidates"], [])

    def test_real_graph_and_chat_project_only_current_location_approved_google_rows(self):
        from test_verifier_location_grounding import VerifierLocationGroundingTests
        case = VerifierLocationGroundingTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        rows = [google_candidate(key) for key in ("fast", "direct", "walk")]
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        tasks = [PlannedTask(agent="location", task="Find centres"), PlannedTask(agent="queue", task="Check queues")]
        with patch.object(orchestrator, "build_model", return_value={}), \
             patch.object(orchestrator, "parse_request", return_value={"parsed_request": case.request, "tasks": tasks}), \
             patch("agents.queue_agent.run", return_value={"agent": "queue", "candidates": rows}), \
             patch("agents.recommendation_agent.ChatOpenAI", side_effect=AssertionError("No extra model")), \
             patch.object(main, "_resolve_session", side_effect=lambda payload: (None, payload)), \
             patch.object(main, "save_trace") as trace:
            response = main.app.test_client().post("/api/chat", json={"message": "How are the queues nearby?"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertLess(body["reply"].index("Fixture walk"), body["reply"].index("Fixture fast"))
        self.assertNotIn("Fixture direct", body["reply"])
        self.assertIn("not observed queue times", body["reply"])
        self.assertEqual([row["centre_id"] for row in body["location_data"]["selected_centres"]], ["walk", "fast"])
        self.assertEqual(body["location_data"]["origin"]["lat"], 1.31234567)  # Existing public contract.
        for value in (body, trace.call_args.args[0], main._session_state_patch(body)):
            text = json.dumps(value)
            for private in ("location_grounding", "recommendation_context", "identity_digest"):
                self.assertNotIn(private, text)
        from services.trace_store import _sanitize
        for value in (_sanitize(trace.call_args.args[0]), main._session_state_patch(body)):
            self.assertNotIn("1.31234567", json.dumps(value))
        case.fixture.response.assert_called_once()
        case.fixture.collect.assert_called_once()

    def test_publication_and_deployment_require_mainbranch_push(self):
        workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/docker.yml").read_text()
        condition = "if: github.event_name == 'push' && github.ref == 'refs/heads/mainbranch'"
        for step in ("Log in to Docker Hub", "Push Backend Docker image", "Push Frontend Docker image"):
            block = re.search(r"- name: " + re.escape(step) + r"\n(.*?)(?=\n      - name:|\n  deploy:)", workflow, re.S)
            self.assertIsNotNone(block)
            self.assertIn(condition, block.group(1))
        self.assertIn("  deploy:\n    " + condition, workflow)
