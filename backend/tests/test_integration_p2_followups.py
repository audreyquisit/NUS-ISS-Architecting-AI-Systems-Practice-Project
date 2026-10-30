"""Offline regressions for empty provider provenance and conservative intent forcing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda

from agents import orchestrator, queue_agent, verifier
from models.orchestration import OrchestrationPlan, ParsedRequest, PlannedTask
from services import google_places_queue_feed as google_feed


class EmptyQueueProvenanceTests(unittest.TestCase):
    def queue_result(self, places, *, model_failure=False, centres=None):
        class ScriptedModel:
            step = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                self.step += 1
                return SimpleNamespace(tool_calls=[{
                    "name": "lookup_queue_records", "args": {"centre_names": []}, "id": "offline-lookup",
                }] if self.step == 1 else [])

        response = Mock()
        response.json.return_value = {"places": places}
        with patch.dict("os.environ", {
            "QUEUE_FEED_PROVIDER": "google_places", "GOOGLE_PLACES_API_KEY": "offline-fixture-key",
        }), patch.object(google_feed.requests, "post", return_value=response), \
             patch.object(queue_agent, "_llm", side_effect=RuntimeError("offline") if model_failure else None,
                          return_value=ScriptedModel()), \
             patch.object(queue_agent, "_rank_candidates", side_effect=AssertionError("No empty ranking call")) as ranking:
            result = queue_agent.run("Check queues", user_request="How are the queues?",
                candidate_centres=centres or ["Maxwell Food Centre"],
                candidate_centre_ids=["maxwell_food_centre"])
        ranking.assert_not_called()
        checked = verifier._verify_worker_result(result, {})
        reply = orchestrator._queue_status_response("How are the queues?", [checked])
        return result, reply

    @staticmethod
    def place(status):
        return {"id": "places/offline-fixture", "displayName": {"text": "Maxwell Food Centre"},
                "rating": 4.0, "userRatingCount": 100, "businessStatus": status}

    def assert_empty_google(self, result, reply):
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["evidence"][0]["source"], "google_places")
        self.assertEqual(result["evidence"][0]["data_sources"], ["google_places"])
        self.assertEqual(result["confidence"], 0)
        self.assertIn("Google Places", reply)
        self.assertIn("No eligible", reply)
        self.assertNotIn("simulated", (reply + str(result["limitations"])).casefold())
        self.assertNotIn("about 0 minutes", reply)

    def test_successful_empty_google_response_does_not_substitute_simulated_rows(self):
        with patch.object(google_feed, "get_mock_queue_feed", side_effect=AssertionError("No simulated substitution")):
            result, reply = self.queue_result([])
        self.assert_empty_google(result, reply)

    def test_closed_only_and_unknown_status_results_keep_google_provenance(self):
        for status in (None, "CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY", "UNKNOWN"):
            with self.subTest(status=status):
                result, reply = self.queue_result([self.place(status)])
                self.assert_empty_google(result, reply)

    def test_location_filtered_google_rows_keep_provider_without_gaining_scope(self):
        result, reply = self.queue_result([self.place("OPERATIONAL")], centres=["Newton Food Centre"])
        self.assert_empty_google(result, reply)
        self.assertNotIn("Maxwell Food Centre", reply)

    def test_deterministic_lookup_after_model_failure_keeps_empty_google_provenance(self):
        result, reply = self.queue_result([self.place("CLOSED_TEMPORARILY")], model_failure=True)
        self.assert_empty_google(result, reply)
        self.assertIn("deterministic provider lookup", str(result["limitations"]))

    def test_missing_key_fallback_reports_simulation_only_when_simulated_rows_used(self):
        simulated = [{"source": "synthetic_queue_mock", "queue_is_mock": True, "queue_minutes": 5}]
        with patch.dict("os.environ", {"GOOGLE_PLACES_API_KEY": ""}), \
             patch.object(google_feed, "get_mock_queue_feed", return_value=simulated), \
             patch.object(google_feed.requests, "post", side_effect=AssertionError("No HTTP without key")):
            self.assertEqual(google_feed.get_live_queue_feed(), simulated)

    def test_unknown_empty_provider_does_not_fabricate_simulated_provenance(self):
        with patch.object(queue_agent, "_agentic_queue_plan", return_value=([], [])):
            result = queue_agent.run("Check queues")
        self.assertEqual(result["evidence"][0]["source"], "unavailable_queue_feed")
        reply = orchestrator._queue_status_response("How are the queues?", [
            verifier._verify_worker_result(result, {})])
        self.assertIn("No verified provider evidence", reply)
        self.assertNotIn("simulated", (reply + str(result["limitations"])).casefold())

    def test_mixed_provider_records_keep_both_limitations_and_actual_sources(self):
        from test_queue_location_integration import google_candidate
        google = {**google_candidate("maxwell_food_centre"), "_agent_candidate_id": "google-0"}
        simulated = {**google, "source": "synthetic_queue_mock", "queue_is_mock": True,
                     "_agent_candidate_id": "mock-0", "stall_id": "source-owned-stall"}
        with patch.object(queue_agent, "_agentic_queue_plan", return_value=([google, simulated], [])), \
             patch.object(queue_agent, "_rank_candidates", return_value=queue_agent.QueueAssessment(
                 reasoning="Offline fixture", confidence=0.5)):
            result = queue_agent.run("Check queues")
        self.assertEqual(result["evidence"][0]["source"], "mixed_queue_feed")
        self.assertEqual(result["evidence"][0]["data_sources"], ["google_places", "synthetic_queue_mock"])
        self.assertTrue(any("popularity" in text for text in result["limitations"]))
        self.assertTrue(any("mock lookup" in text for text in result["limitations"]))


class ConservativeIntentForcingTests(unittest.TestCase):
    def parse(self, message, *, context=None, intent="other"):
        plan = OrchestrationPlan(request=ParsedRequest(intent=intent), tasks=[
            PlannedTask(agent=agent, task="Untrusted proposed task") for agent in ("location", "dietary", "queue")
        ])
        response = Mock(return_value=plan)
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(response)
        result = orchestrator.parse_request({"user_request": message, "model": model,
                                            "conversation_context": context or {}})
        self.assertLessEqual(response.call_count, 1)
        return result

    def test_ordinary_terminology_and_programming_questions_keep_nonfood_intent(self):
        for message in ("What does the word queue mean?", "Define queue.",
                        "Explain the term food queue.", "How does a computer queue work?",
                        "Show me a Python queue example.", "How do I clear my printer queue?",
                        "Show me the definition of a food queue."):
            with self.subTest(message=message):
                result = self.parse(message)
                self.assertEqual(result["parsed_request"]["intent"], "other")
                self.assertEqual(result["tasks"], [])

    def test_ambiguous_keywords_do_not_override_nonfood_interpretation(self):
        for message in ("Queue?", "Waiting time", "A different queue"):
            with self.subTest(message=message):
                result = self.parse(message)
                self.assertEqual(result["parsed_request"]["intent"], "other")
                self.assertEqual(result["tasks"], [])

    def test_explicit_food_and_named_centre_requests_still_recover_missed_intent(self):
        for message in ("Find chicken rice near Bugis.", "I want halal food under $8 near Bugis.",
                        "How are the queues at Maxwell Food Centre?", "Find queues at Chinatown Complex.",
                        "How is the weather at Chinatown Complex?"):
            with self.subTest(message=message):
                result = self.parse(message)
                self.assertEqual(result["parsed_request"]["intent"], "food_discovery")
                self.assertTrue(result["tasks"])

    def test_relevant_deictic_queue_followup_retains_food_context(self):
        context = {"turns": [
            {"role": "user", "content": "Find chicken rice at Maxwell Food Centre."},
            {"role": "assistant", "content": "Here are the food options at Maxwell Food Centre."},
        ]}
        result = self.parse("How are the queues there?", context=context)
        self.assertEqual(result["parsed_request"]["intent"], "food_discovery")
        self.assertIn("queue", [task.agent for task in result["tasks"]])
        ordinary = self.parse("What does the word queue mean?", context=context)
        self.assertEqual(ordinary["parsed_request"]["intent"], "other")
        self.assertEqual(ordinary["tasks"], [])

    def test_pending_food_offer_and_location_answer_keep_existing_context_gate(self):
        for message, assistant in (("Yes", "Would you like me to show other food options?"),
                                   ("Bugis", "Where are you starting from?")):
            with self.subTest(message=message):
                result = self.parse(message, context={"turns": [
                    {"role": "user", "content": "Find chicken rice."},
                    {"role": "assistant", "content": assistant},
                ]})
                self.assertEqual(result["parsed_request"]["intent"], "food_discovery")

    def test_valid_food_intent_is_not_downgraded_by_ambiguous_wording(self):
        result = self.parse("Queue?", intent="food_discovery")
        self.assertEqual(result["parsed_request"]["intent"], "food_discovery")
        self.assertIn("queue", [task.agent for task in result["tasks"]])

    def test_actual_chat_graph_does_not_dispatch_workers_for_definition_question(self):
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        response = Mock(return_value=OrchestrationPlan(request=ParsedRequest(intent="other"), tasks=[]))
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(response)
        with patch.object(orchestrator, "build_model", return_value={"model": model}), \
             patch.object(orchestrator, "location_worker", side_effect=AssertionError("No Location")) as location, \
             patch.object(orchestrator, "worker", side_effect=AssertionError("No worker")) as worker, \
             patch("agents.recommendation_agent.ChatOpenAI", side_effect=AssertionError("No Recommendation")), \
             patch.object(main, "_resolve_session", side_effect=lambda payload: (None, payload)), \
             patch.object(main, "save_trace"):
            api = main.app.test_client().post("/api/chat", json={"message": "What does the word queue mean?"})
        self.assertEqual(api.status_code, 200)
        self.assertEqual(api.get_json()["parsed_request"]["intent"], "other")
        self.assertEqual(api.get_json()["worker_plan"], [])
        location.assert_not_called()
        worker.assert_not_called()
        response.assert_called_once()
