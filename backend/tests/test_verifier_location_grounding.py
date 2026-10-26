"""Offline verifier trust boundaries with current Location evidence and real local matchers."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from agents import location_grounding as grounding, orchestrator, verifier
from models.location_models import LocationDecision
from models.orchestration import PlannedTask
from services.activity_log import begin_activity_capture, end_activity_capture
from test_location_grounded_reasoning import comparative_reason
import test_location_grounding_handoff as handoff_fixtures


class VerifierLocationGroundingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = handoff_fixtures.LocationGroundingHandoffTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.request = {"intent": "food_discovery"}
        self.menu = [{
            "menu_item_id": item.centre_id + ":rice", "stall_id": item.centre_id + ":stall",
            "hawker_centre_id": item.centre_id, "hawker_centre": item.centre_name,
            "stall_name": "Fixture Rice Stall " + item.centre_id,
            "menu_item_name": "Chicken rice", "categories": ["Chinese", "Rice"],
            "dietary_tags": [], "price_sgd_min": 4.0, "price_sgd_max": 6.0,
            "price_label": "$4–6", "is_mock": True,
        } for item in self.fixture.routes]
        self.fixture.patches.enter_context(patch.object(verifier, "load_hawker_centres", return_value=self.fixture.catalogue))
        self.fixture.patches.enter_context(patch("services.hawker_service._menu_catalogue", return_value=tuple(self.menu)))
        self.fixture.patches.enter_context(patch.object(verifier, "get_menu_item_record", side_effect=lambda key:
            next((item for item in self.menu if item["menu_item_id"] == key), None)))

    def execute(self):
        return self.fixture.execute()

    def verify(self, decision, bundle=None, *, request=None, reused=False, workers=()):
        return verifier.verify_results(
            [orchestrator._result_envelope("location", decision), *workers],
            self.request if request is None else request, location_grounding=bundle,
            request_id=self.fixture.query.request_id, location_reused=reused,
        )

    def checked(self, decision, bundle=None, **kwargs):
        return self.verify(decision, bundle, **kwargs)[0]["payload"]

    def worker(self, agent="dietary", items=None):
        return {"agent": agent, "candidates": copy.deepcopy(self.menu if items is None else items)}

    @staticmethod
    def route(decision, centre):
        return next(item for item in decision["logistics"] if item["centre_id"] == centre)

    @staticmethod
    def choice(checked, centre):
        return next(item for item in checked["selected_centres"] if item["centre_id"] == centre)

    @staticmethod
    def joined(results):
        return next(item["candidates"] for item in results if item["agent"] == "candidate_join")

    def queue_worker(self, ids=("walk", "fast", "direct")):
        from test_queue_guardrails import _candidate
        self.fixture.patches.enter_context(patch.object(verifier, "get_menu_stall_record", side_effect=lambda key:
            next((item for item in self.menu if item["stall_id"] == key), None)))
        return {"agent": "queue", "candidates": [_candidate(stall_id=key + ":stall", centre_id=key) for key in ids]}

    def test_queue_scope_excludes_rejected_and_comparison_only_centres(self):
        decision, bundle = self.execute()
        decision["selected_centres"] = decision["selected_centres"][:1]
        results = self.verify(decision, bundle, workers=[self.queue_worker()])
        queue = next(row for row in results if row["agent"] == "queue")
        self.assertEqual([row["hawker_centre_id"] for row in queue["candidates"]], ["walk"])
        reply = orchestrator._queue_status_response("How are the queues?", results)
        self.assertIn("Fixture walk", reply)
        self.assertNotIn("Fixture fast", reply)
        self.assertNotIn("Fixture direct", reply)

    def test_empty_location_scope_rejects_queue_in_verifier_and_synthesis(self):
        decision, bundle = self.execute()
        decision["selected_centres"] = []
        queue = self.queue_worker()
        results = self.verify(decision, bundle, workers=[queue])
        self.assertEqual(next(row for row in results if row["agent"] == "queue")["candidates"], [])
        # Also defend the deterministic renderer independently of worker filtering.
        reply = orchestrator._queue_status_response("How are the queues?", [results[0], {**queue, "verified": True}])
        self.assertNotIn("Fixture walk", reply)
        self.assertNotIn("Untrusted centre label", reply)

    def test_mixed_food_queue_rejection_does_not_shortcut_food_eligibility(self):
        decision, bundle = self.execute()
        request = {**self.request, "dietary_preferences": ["vegetarian"]}
        results = self.verify(decision, bundle, request=request, workers=[self.worker(), self.queue_worker()])
        self.assertEqual(self.joined(results), [])
        recommendation = Mock(return_value="No verified food options")
        with patch("agents.recommendation_agent.run", recommendation):
            reply = orchestrator.synthesizer({"user_request": "Find vegetarian food and compare queues",
                "parsed_request": request, "verified_results": results})["final_recommendation"]
        self.assertNotIn("SIMULATED QUEUE DATA", reply)
        recommendation.assert_not_called()

    def test_mixed_food_queue_success_uses_only_food_eligible_stalls_without_model_call(self):
        decision, bundle = self.execute()
        results = self.verify(decision, bundle, workers=[self.worker(), self.queue_worker()])
        joined = next(row for row in results if row["agent"] == "candidate_join")
        joined["candidates"] = [row for row in joined["candidates"] if row["hawker_centre_id"] == "fast"]
        with patch("agents.recommendation_agent.run", side_effect=AssertionError("No extra call")):
            reply = orchestrator.synthesizer({"user_request": "Find chicken rice and compare queues",
                "parsed_request": {**self.request, "menu_item_preferences": ["chicken rice"]},
                "verified_results": results})["final_recommendation"]
        self.assertIn("Fixture fast", reply)
        self.assertNotIn("Fixture walk", reply)

    def test_unscoped_pure_and_named_stall_queue_requests_survive(self):
        for request, message in (({"intent": "food_discovery"}, "How are the queues?"),
                ({"intent": "food_discovery", "stall_name_preferences": ["Fixture Rice Stall walk"]},
                 "What is the queue for Fixture Rice Stall walk?")):
            with self.subTest(request=request):
                results = verifier.verify_results([self.queue_worker(("walk", "fast"))], request)
                reply = orchestrator.synthesizer({"parsed_request": request, "user_request": message,
                    "verified_results": results})["final_recommendation"]
                self.assertIn("Fixture walk", reply)
                self.assertIn("SIMULATED QUEUE DATA", reply)
                if request.get("stall_name_preferences"):
                    self.assertNotIn("Fixture fast", reply)

    def test_exclusions_require_food_eligibility_before_queue_reply(self):
        decision, bundle = self.execute()
        request = {**self.request, "excluded_menu_item_ids": [row["menu_item_id"] for row in self.menu]}
        results = self.verify(decision, bundle, request=request, workers=[self.worker(), self.queue_worker()])
        self.assertEqual(self.joined(results), [])
        with patch("agents.recommendation_agent.run", side_effect=AssertionError("No extra call")):
            reply = orchestrator.synthesizer({"parsed_request": request,
                "user_request": "Other options with short queues", "verified_results": results})["final_recommendation"]
        self.assertNotIn("SIMULATED QUEUE DATA", reply)

    def test_queue_hard_travel_limit_without_location_evidence_fails_closed(self):
        for intent in ("queue", "food_discovery"):
            with self.subTest(intent=intent):
                results = verifier.verify_results([self.queue_worker()], {"intent": intent, "max_travel_time_min": 10})
                self.assertEqual(next(row for row in results if row["agent"] == "queue")["candidates"], [])

    def test_farther_unapproved_queue_cannot_shortcut_approval(self):
        decision, bundle = self.execute()
        self.menu[2]["menu_item_name"] = "Exclusive dish"
        self.fixture.matches["walk"] = self.fixture.matches["fast"] = self.fixture.matches["unknown"] = []
        self.route(decision, "direct")["travel"]["total_duration_min"] = 40
        state = self.fixture.state(decision, bundle)
        state["parsed_request"]["menu_item_preferences"] = ["Exclusive dish"]
        state.update(orchestrator.expand_menu_search(state))
        results = self.verify(state["location_result"], state["location_grounding"],
            request=state["parsed_request"], workers=[self.worker(), self.queue_worker(("direct",))])
        queue = next(row for row in results if row["agent"] == "queue")
        self.assertEqual(queue["candidates"], [])
        self.assertEqual(self.joined(results), [])
        with patch("agents.recommendation_agent.run", side_effect=AssertionError("No extra call")):
            reply = orchestrator.synthesizer({"parsed_request": state["parsed_request"],
                "user_request": "Find rice with short queues", "verified_results": results})["final_recommendation"]
        self.assertIn("Would you like", reply)
        self.assertNotIn("SIMULATED QUEUE DATA", reply)

    def test_malformed_route_mode_selection_id_and_expansion_containers_fail_closed(self):
        decision, bundle = self.execute()
        for malformed in ([], {}):
            with self.subTest(value=malformed):
                changed = copy.deepcopy(decision)
                self.route(changed, "walk")["travel"]["mode"] = malformed
                self.assertNotIn("walk", {row["centre_id"] for row in self.checked(changed)["selected_centres"]})
                changed = copy.deepcopy(decision)
                changed["selected_centres"][0]["centre_id"] = malformed
                self.assertEqual([row["centre_id"] for row in self.checked(changed, reused=True)["selected_centres"]], ["fast"])
                self.assertEqual([row["centre_id"] for row in self.checked(changed, bundle)["selected_centres"]], ["fast"])
                changed = copy.deepcopy(decision)
                changed["search_expansion"] = {"outcome": malformed}
                self.assertEqual(self.checked(changed)["search_expansion"], {})
                changed["search_expansion"] = {"outcome": "expanded_match_found",
                    "additional_centres": [{"centre_id": malformed}]}
                self.assertEqual(self.checked(changed)["search_expansion"]["additional_centres"], [])

    def test_catalogue_only_and_fresh_route_wording_are_distinct(self):
        decision, bundle = self.execute()
        for reused in (True, False):
            results = self.verify(decision, bundle, reused=reused)
            with patch("services.location_service.load_hawker_centres", return_value=self.fixture.catalogue):
                reply = orchestrator.synthesizer({"parsed_request": self.request,
                    "verified_results": results})["final_recommendation"]
            if reused:
                self.assertIn("Catalogue", reply)
                self.assertNotIn("route-checked", reply)
                self.assertNotIn("closest", reply)
                self.assertNotIn("Walking distance and time", reply)
            else:
                self.assertIn("route-checked", reply)
                self.assertIn("15 min", reply)

    def test_cached_queue_summary_does_not_claim_proximity(self):
        decision, bundle = self.execute()
        results = self.verify(decision, bundle, reused=True, workers=[self.queue_worker()])
        reply = orchestrator.synthesizer({"parsed_request": self.request,
            "user_request": "How are the queues?", "verified_results": results})["final_recommendation"]
        self.assertIn("Fixture walk", reply)
        self.assertNotIn("near Fixture origin", reply)
        self.assertNotIn("Walking distance and time", reply)

    def test_actual_graph_empty_approval_hard_limit_cannot_return_queue_recommendations(self):
        decision, _ = self.execute()
        request = {**self.request, "max_travel_time_min": 10, "menu_item_preferences": ["rice"]}
        tasks = [PlannedTask(agent="dietary", task="Find food"), PlannedTask(agent="queue", task="Compare queues")]
        queue = self.queue_worker()
        with patch.object(orchestrator, "build_model", return_value={}), \
             patch.object(orchestrator, "parse_request", return_value={"parsed_request": request, "tasks": tasks}), \
             patch("agents.dietary_agent.run", return_value=self.worker()), \
             patch("agents.queue_agent.run", return_value=queue), \
             patch("agents.recommendation_agent.run", side_effect=AssertionError("No extra call")):
            result = orchestrator.build_workflow().invoke({"request_id": "new-request",
                "user_request": "Find rice nearby and compare queues", "tasks": [], "agent_results": [],
                "conversation_context": {"state": {"location_result": decision}}})
        location = next(row["payload"] for row in result["verified_results"] if row["agent"] == "location")
        self.assertTrue(result["location_reused"])
        self.assertEqual(location["selected_centres"], [])
        self.assertEqual(self.joined(result["verified_results"]), [])
        self.assertNotIn("SIMULATED QUEUE DATA", result["final_recommendation"])
        self.assertNotIn("Fixture walk", result["final_recommendation"])
        self.assertIn("hard travel-time limit", result["final_recommendation"])
        self.fixture.response.assert_called_once()  # Prior turn only; no fresh ranking on reuse.

    def test_actual_no_match_expansion_reused_in_graph_has_no_current_travel_claims(self):
        decision, bundle = self.execute()
        for key in self.fixture.matches:
            self.fixture.matches[key] = []
        prior = self.fixture.state(decision, bundle)
        prior.update(orchestrator.expand_menu_search(prior))
        self.assertEqual(prior["location_result"]["search_expansion"]["outcome"], "no_match_in_routed_pool")
        self.assertEqual(prior["location_result"]["search_expansion"]["route_checked_centre_count"], 4)
        request = {**self.request, "dietary_preferences": ["vegetarian"]}
        tasks = [PlannedTask(agent="dietary", task="Find food"), PlannedTask(agent="queue", task="Check queues")]
        queue = self.queue_worker()
        for message in ("Find vegetarian food", "Find vegetarian food and compare queues"):
            with self.subTest(message=message), \
                 patch.object(orchestrator, "build_model", return_value={}), \
                 patch.object(orchestrator, "parse_request", return_value={"parsed_request": request, "tasks": tasks}), \
                 patch("agents.dietary_agent.run", return_value=self.worker()), \
                 patch("agents.queue_agent.run", return_value=queue), \
                 patch("agents.recommendation_agent.run", side_effect=AssertionError("No additional model call")):
                result = orchestrator.build_workflow().invoke({"request_id": "reused-expansion",
                    "user_request": message, "tasks": [], "agent_results": [],
                    "conversation_context": {"state": {"location_result": prior["location_result"]}}})
                location = next(row["payload"] for row in result["verified_results"] if row["agent"] == "location")
                self.assertTrue(result["location_reused"])
                self.assertEqual(location["logistics"], [])
                self.assertEqual(location["search_expansion"], {})
                self.assertEqual(self.joined(result["verified_results"]), [])
                reply = result["final_recommendation"]
                self.assertIn("couldn't verify food options", reply)
                for unsupported in ("route-checked", "30 minutes", "Fixture origin", "SIMULATED QUEUE DATA"):
                    self.assertNotIn(unsupported, reply)
                self.assertIsNone(result["location_grounding"])
                self.assertNotIn(bundle.execution_id, json.dumps(result["verified_results"]))
                LocationDecision.model_validate({key: location[key] for key in LocationDecision.model_fields if key in location})
        self.fixture.response.assert_called_once()  # Prior turn only.
        self.fixture.collect.assert_called_once()

    def test_farther_expansion_requires_current_route_support(self):
        decision, bundle = self.execute()
        self.menu[2]["menu_item_name"] = "Exclusive dish"
        self.fixture.matches["walk"] = self.fixture.matches["fast"] = self.fixture.matches["unknown"] = []
        self.route(decision, "direct")["travel"]["total_duration_min"] = 40
        prior = self.fixture.state(decision, bundle)
        prior["parsed_request"]["menu_item_preferences"] = ["Exclusive dish"]
        prior.update(orchestrator.expand_menu_search(prior))
        request = prior["parsed_request"]
        for kind in ("fresh", "reused", "catalogue", "missing_routes", "missing_farther_route"):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(prior["location_result"])
                current_request = dict(request)
                if kind == "missing_routes":
                    changed["logistics"] = []
                elif kind == "missing_farther_route":
                    changed["logistics"] = [row for row in changed["logistics"] if row["centre_id"] != "direct"]
                elif kind == "catalogue":
                    current_request["search_scope_mode"] = "global"
                results = self.verify(changed, prior["location_grounding"], request=current_request,
                    reused=kind == "reused", workers=[self.worker()])
                location = results[0]["payload"]
                self.assertEqual(self.joined(results), [])
                with patch("agents.recommendation_agent.run", side_effect=AssertionError("No additional model call")):
                    reply = orchestrator.synthesizer({"parsed_request": current_request,
                        "user_request": "Find Exclusive dish", "verified_results": results})["final_recommendation"]
                if kind == "fresh":
                    self.assertEqual(location["search_expansion"]["outcome"], "farther_match_requires_approval")
                    self.assertIn("Fixture direct", reply)
                    self.assertIn("40 minutes", reply)
                    self.assertIn("Would you like", reply)
                else:
                    self.assertEqual(location["search_expansion"], {})
                    self.assertIn("couldn't verify food options", reply)
                    for unsupported in ("within 30", "40 minutes", "nearest catalogue match"):
                        self.assertNotIn(unsupported, reply)
        self.fixture.response.assert_called_once()
        self.fixture.collect.assert_called_once()

    def test_fresh_no_match_expansion_counts_only_surviving_verified_routes(self):
        decision, bundle = self.execute()
        for key in self.fixture.matches:
            self.fixture.matches[key] = []
        state = self.fixture.state(decision, bundle)
        state["parsed_request"]["menu_item_preferences"] = ["Unavailable dish"]
        state.update(orchestrator.expand_menu_search(state))
        for remove_route in (False, True):
            with self.subTest(remove_route=remove_route):
                changed = copy.deepcopy(state["location_result"])
                if remove_route:
                    self.route(changed, "direct")["travel"]["source"] = "unavailable"
                results = self.verify(changed, state["location_grounding"], request=state["parsed_request"], workers=[self.worker()])
                count = 3 if remove_route else 4
                self.assertEqual(results[0]["payload"]["search_expansion"]["route_checked_centre_count"], count)
                self.assertEqual(self.joined(results), [])
                with patch("agents.recommendation_agent.run", side_effect=AssertionError("No additional model call")):
                    reply = orchestrator.synthesizer({"parsed_request": state["parsed_request"],
                        "user_request": "Find Unavailable dish", "verified_results": results})["final_recommendation"]
                self.assertIn(f"{count} route-checked centres within 30 minutes", reply)

    def test_nontravel_dietary_limitation_survives_without_cached_travel_metadata(self):
        decision, _ = self.execute()
        decision["search_expansion"] = {"outcome": "dietary_verification_unavailable",
            "route_checked_centre_count": 4, "search_horizon_min": 30}
        for diets in (["vegetarian"], []):
            with self.subTest(diets=diets):
                request = {**self.request, "dietary_preferences": diets, "menu_item_preferences": ["Unavailable dish"]}
                results = self.verify(decision, reused=True, request=request, workers=[self.worker()])
                self.assertEqual(results[0]["payload"]["search_expansion"],
                    {"outcome": "dietary_verification_unavailable"} if diets else {})
                with patch("agents.recommendation_agent.run", side_effect=AssertionError("No additional model call")):
                    reply = orchestrator.synthesizer({"parsed_request": request,
                        "user_request": "Find food", "verified_results": results})["final_recommendation"]
                self.assertNotIn("route-checked", reply)
                self.assertNotIn("30 minutes", reply)
                self.assertIn("can't verify that dietary restriction" if diets else "couldn't verify food options", reply)

    def test_valid_model_selection_preserves_nonfastest_order_rationale_fit_and_confidence(self):
        self.fixture.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"],
        )]
        decision, bundle = self.execute()
        before = copy.deepcopy(decision)
        checked = self.checked(decision, bundle)
        self.assertEqual(checked["selected_centres"], decision["selected_centres"])
        self.assertEqual(checked["confidence"], 0.97)
        self.assertIn("Location model-ranked", checked["reasoning"])
        self.assertIn("not calibrated", " ".join(checked["limitations"]))
        self.assertEqual(decision, before)
        self.fixture.response.assert_called_once()
        LocationDecision.model_validate({key: checked[key] for key in LocationDecision.model_fields if key in checked})

    def test_menu_filter_refreshes_summary_without_retaining_removed_centre(self):
        decision, bundle = self.execute()
        self.fixture.matches["fast"] = []
        state = self.fixture.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        checked = self.checked(state["location_result"], state["location_grounding"])
        self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["walk"])
        self.assertNotIn("Fixture fast", checked["reasoning"])
        self.assertEqual(checked["confidence"], 0.5)
        self.assertNotIn("initial_location_reasoning", checked["search_expansion"])
        self.assertIn("policy", " ".join(checked["limitations"]))

    def test_menu_supplements_have_deterministic_provenance_and_policy_fit(self):
        decision, bundle = self.execute()
        self.fixture.matches["fast"] = self.fixture.matches["walk"] = []
        state = self.fixture.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        checked = self.checked(state["location_result"], state["location_grounding"])
        self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["unknown", "direct"])
        self.assertTrue(all(item["logistics_fit"] == 0.5 and "Deterministic menu-expansion" in item["rationale"]
                            for item in checked["selected_centres"]))
        self.assertEqual(checked["confidence"], 0.5)
        self.assertNotIn("Location model-ranked", checked["reasoning"])

    def test_unselected_comparison_supports_claim_but_never_food_scope(self):
        self.fixture.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        decision, bundle = self.execute()
        results = self.verify(decision, bundle, workers=[self.worker()])
        checked = results[0]["payload"]
        self.assertEqual(self.choice(checked, "walk"), decision["selected_centres"][0])
        self.assertIn("Fixture direct", self.choice(checked, "walk")["rationale"])
        self.assertEqual({item["centre_id"] for item in checked["logistics"]}, {"walk", "fast"})
        self.assertEqual({item["hawker_centre_id"] for item in self.joined(results)}, {"walk", "fast"})

    def test_invalid_references_false_comparisons_and_captured_facts_lose_authority(self):
        decision, bundle = self.execute()
        original = bundle.selections[0]
        reasons = [
            original.reasons[0].model_copy(update={"evidence_refs": ("absent:primary:total_duration_min",)}),
            grounding._Reason(code="SHORTER_DURATION", comparison_centre_id="fast", evidence_refs=(
                "walk:primary:total_duration_min", "fast:primary:total_duration_min")),
        ]
        for reason in reasons:
            with self.subTest(code=reason.code):
                forged = bundle.model_copy(update={"selections": (
                    original.model_copy(update={"reasons": (reason,)}), bundle.selections[1])})
                checked = self.checked(decision, forged)
                self.assertEqual(self.choice(checked, "walk")["logistics_fit"], 0)
                self.assertIn("proof was missing or rejected", self.choice(checked, "walk")["rationale"])
                self.assertEqual(checked["confidence"], 0.5)
        captured = next(item for item in bundle.routes if item.centre_id == "walk")
        altered = captured.model_copy(update={"facts": (
            captured.facts[0].model_copy(update={"value": 1}), *captured.facts[1:])})
        forged = bundle.model_copy(update={"routes": tuple(altered if item.centre_id == "walk" else item for item in bundle.routes)})
        self.assertEqual(self.choice(self.checked(decision, forged), "walk")["logistics_fit"], 0)

    def test_changed_comparison_invalidates_dependent_claim_only(self):
        self.fixture.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        decision, bundle = self.execute()
        self.route(decision, "direct")["travel"]["walking_distance_m"] = 50
        checked = self.checked(decision, bundle)
        self.assertEqual(self.choice(checked, "walk")["logistics_fit"], 0)
        self.assertEqual(self.choice(checked, "fast")["logistics_fit"], 0.9)

    def test_unsupported_prose_and_route_displays_are_not_verified_claims(self):
        decision, bundle = self.execute()
        decision["selected_centres"][0]["rationale"] = "Fully sheltered and wheelchair accessible"
        decision["reasoning"] = "All routes are fully sheltered"
        decision["limitations"] = ["Wheelchair access confirmed"]
        self.route(decision, "walk")["travel"].update(
            itinerary_summary="Fully sheltered", walking_distance_display="Wheelchair accessible")
        checked = self.checked(decision, bundle)
        text = json.dumps(checked).casefold()
        self.assertNotIn("sheltered", text)
        self.assertNotIn("wheelchair", text)
        self.assertEqual(self.choice(checked, "walk")["logistics_fit"], 0)

    def test_conflicting_duplicates_are_rejected_before_bad_copy_filtering_in_both_orders(self):
        decision, bundle = self.execute()
        original = self.route(decision, "walk")
        for section, field, value in (("travel", "total_duration_min", 60), ("travel", "transfers", 2),
                                      ("coordinates", "lat", 1.4), (None, "route_id", "other"),
                                      ("travel", "source", "unavailable")):
            duplicate = copy.deepcopy(original)
            (duplicate[section] if section else duplicate)[field] = value
            for pair in ((original, duplicate), (duplicate, original)):
                with self.subTest(field=field, invalid_first=pair[0] is duplicate):
                    current = copy.deepcopy(decision)
                    current["logistics"] = [item for item in current["logistics"] if item["centre_id"] != "walk"] + list(pair)
                    checked = self.checked(current, bundle)
                    self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["fast"])

    def test_numeric_equivalent_duplicates_preserve_grounding_in_both_orders(self):
        decision, bundle = self.execute()
        original = self.route(decision, "walk")
        duplicate = copy.deepcopy(original)
        duplicate["travel"]["walking_distance_m"] = 100
        for pair in ((original, duplicate), (duplicate, original)):
            current = copy.deepcopy(decision)
            current["logistics"] = [item for item in current["logistics"] if item["centre_id"] != "walk"] + list(pair)
            self.assertEqual(self.checked(current, bundle)["selected_centres"], decision["selected_centres"])

    def test_invalid_measurements_and_destinations_fail_closed(self):
        decision, bundle = self.execute()
        for metric in ("total_duration_min", "walking_distance_m", "walking_time_min", "transfers"):
            for value in (-1, False, True, float("nan"), float("inf"), "5"):
                with self.subTest(metric=metric, value=value):
                    current = copy.deepcopy(decision)
                    self.route(current, "walk")["travel"][metric] = value
                    self.assertNotIn("walk", [item["centre_id"] for item in self.checked(current, bundle)["selected_centres"]])
        for coords in ({"lat": float("nan"), "lng": 103.8}, {"lat": True, "lng": 103.8}):
            current = copy.deepcopy(decision)
            self.route(current, "walk")["coordinates"] = coords
            self.assertEqual([item["centre_id"] for item in self.checked(current, bundle)["selected_centres"]], ["fast"])

    def test_unsupported_modes_sources_missing_duration_and_fractional_transfers(self):
        decision, bundle = self.execute()
        for field, value in (("mode", "flight"), ("source", "unavailable"),
                             ("total_duration_min", None), ("transfers", 1.5)):
            current = copy.deepcopy(decision)
            self.route(current, "walk")["travel"][field] = value
            self.assertEqual([item["centre_id"] for item in self.checked(current, bundle)["selected_centres"]], ["fast"])

    def test_hard_limit_unknown_routes_and_zero_limit(self):
        decision, bundle = self.execute()
        checked = self.checked(decision, bundle, request={"intent": "food_discovery", "max_travel_time_min": 10})
        self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["fast"])
        self.assertEqual(checked["confidence"], 0.5)
        for limit in (0, -1, True, float("nan")):
            checked = self.checked(decision, bundle, request={"intent": "food_discovery", "max_travel_time_min": limit})
            self.assertEqual(checked["selected_centres"], [])
            self.assertEqual(checked["confidence"], 0)
        decision["selected_centres"] = [decision["selected_centres"][1]]
        self.route(decision, "fast")["travel"]["total_duration_min"] = 0
        self.assertEqual(len(self.checked(decision, request={"intent": "food_discovery", "max_travel_time_min": 0})["selected_centres"]), 1)

    def test_missing_metrics_stay_unknown_and_known_walking_time_is_approximate(self):
        self.fixture.response.return_value.selected_centres[0].centre_id = "unknown"
        self.fixture.response.return_value.selected_centres[0].route_id = "unknown:primary"
        from agents.location_agent import _RouteReason
        self.fixture.response.return_value.selected_centres[0].reasons = [_RouteReason(
            code="METRIC_UNAVAILABLE", evidence_refs=["unknown:primary:walking_distance_m", "unknown:primary:transfers"])]
        decision, bundle = self.execute()
        checked = self.checked(decision, bundle)
        travel = self.route(checked, "unknown")["travel"]
        self.assertIsNone(travel["walking_distance_m"])
        self.assertIsNone(travel["walking_time_min"])
        self.assertIsNone(travel["transfers"])
        self.assertIn("unavailable", self.choice(checked, "unknown")["rationale"])
        self.assertIn("approximate", " ".join(self.route(checked, "fast")["limitations"]))

    def test_location_fallback_keeps_deterministic_policy_rationale_and_fit(self):
        self.fixture.response.side_effect = TimeoutError()
        decision, bundle = self.execute()
        checked = self.checked(decision, bundle)
        self.assertEqual(checked["selected_centres"], decision["selected_centres"])
        self.assertEqual(checked["confidence"], 0.5)
        self.assertIn("Location deterministic fallback", checked["reasoning"])
        self.fixture.response.assert_called_once()

    def test_missing_serialized_wrong_request_and_invalid_private_proof_are_conservative(self):
        decision, bundle = self.execute()
        for proof in (None, bundle.model_dump(), {"validated": True},
                      bundle.model_copy(update={"request_id": "wrong-request"}),
                      bundle.model_copy(update={"confidence": float("nan")})):
            with self.subTest(proof_type=type(proof).__name__):
                checked = self.checked(decision, proof)
                self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["walk", "fast"])
                self.assertTrue(all(item["logistics_fit"] == 0 for item in checked["selected_centres"]))
                self.assertEqual(checked["confidence"], 0.5)
                self.assertNotIn("Location model-ranked", checked["reasoning"])

    def test_cached_reuse_drops_travel_claims_and_fails_closed_with_hard_limit(self):
        decision, bundle = self.execute()
        for current in (decision, {"selected_centres": decision["selected_centres"], "logistics": []}):
            checked = self.checked(current, bundle, reused=True)
            self.assertEqual(len(checked["selected_centres"]), 2)
            self.assertEqual(checked["logistics"], [])
            self.assertTrue(all(item["logistics_fit"] == 0 for item in checked["selected_centres"]))
            results = self.verify(current, bundle, reused=True, workers=[self.worker()],
                                  request={"intent": "food_discovery", "max_travel_time_min": 20})
            self.assertEqual(results[0]["payload"]["selected_centres"], [])
            self.assertEqual(self.joined(results), [])

    def test_catalogue_region_global_and_named_centre_paths_remain_available(self):
        decision, _ = self.execute()
        for request in ({"intent": "centre_details"}, {"intent": "food_discovery", "search_scope_mode": "global"},
                        {"intent": "food_discovery", "search_scope_mode": "region", "requested_area": "east"},
                        {"intent": "food_discovery", "target_centre_text": "Fixture walk"}):
            current = copy.deepcopy(decision)
            current.update(logistics=[], origin=None)
            checked = self.checked(current, request=request)
            self.assertEqual(len(checked["selected_centres"]), 2)
            self.assertEqual(checked["logistics"], [])
            self.assertIn("catalogue only", checked["reasoning"])

    def test_named_directions_need_eligible_routes_but_no_ranking_model_proof(self):
        decision, _ = self.execute()
        checked = self.checked(decision, request={"intent": "directions", "target_centre_text": "Fixture walk"})
        self.assertEqual(len(checked["selected_centres"]), 2)
        self.assertTrue(all(item["logistics_fit"] == 0 for item in checked["selected_centres"]))

    def test_source_food_identity_and_variable_starting_price_semantics_are_preserved(self):
        decision, bundle = self.execute()
        worker = self.worker()
        worker["candidates"][0].update(stall_name="Forged", menu_item_name="Free food", price_sgd_min=0.01)
        results = self.verify(decision, bundle, workers=[worker, self.worker("budget")], request={
            "intent": "food_discovery", "budget_amount": 5.0, "menu_item_preferences": ["chicken", "rice"],
        })
        joined = self.joined(results)
        self.assertEqual(len(joined), 2)
        self.assertTrue(all(item["price_sgd_min"] == 4.0 and item["price_sgd_max"] == 6.0 for item in joined))
        self.assertEqual(next(item for item in joined if item["hawker_centre_id"] == "fast")["stall_name"], self.menu[0]["stall_name"])

    def test_real_local_matcher_rejects_dish_stall_exclusion_cuisine_diet_and_budget(self):
        decision, bundle = self.execute()
        for fields in ({"menu_item_preferences": ["laksa"]}, {"stall_name_preferences": ["Different Stall"]},
                       {"excluded_menu_item_ids": [item["menu_item_id"] for item in self.menu]},
                       {"cuisine_preferences": ["Western"]}, {"dietary_preferences": ["vegetarian"]},
                       {"dietary_preferences": ["halal"]}, {"budget_amount": 3.0}):
            with self.subTest(fields=fields):
                results = self.verify(decision, bundle, workers=[self.worker()], request={**self.request, **fields})
                self.assertEqual(self.joined(results), [])
                self.assertEqual(results[1]["candidates"], [])
                self.assertEqual(results[1]["payload"].get("candidates"), [])

    def test_empty_approval_never_calls_unrestricted_matchers(self):
        decision, bundle = self.execute()
        decision["logistics"] = []
        with patch.object(verifier, "find_dietary_stalls", side_effect=AssertionError("No unrestricted matching")), \
             patch.object(verifier, "find_stalls_within_budget", side_effect=AssertionError("No unrestricted matching")):
            results = self.verify(decision, bundle, workers=[self.worker(), self.worker("budget")])
        self.assertEqual(self.joined(results), [])
        self.assertTrue(all(item["candidates"] == [] for item in results))

    def test_food_scope_is_independent_of_envelope_order_and_cannot_be_reintroduced(self):
        decision, bundle = self.execute()
        location = orchestrator._result_envelope("location", decision)
        for ordered in ([self.worker(), location], [location, self.worker()]):
            results = verifier.verify_results(ordered, self.request, location_grounding=bundle,
                request_id=self.fixture.query.request_id)
            self.assertEqual({item["hawker_centre_id"] for item in self.joined(results)}, {"walk", "fast"})
        results = verifier.verify_results([self.worker()], self.request)
        self.assertEqual(self.joined(results), [])

    def test_canonical_food_matching_accepts_normalized_cuisine_and_inferred_diet(self):
        for item in self.menu:
            item.update(categories=["South-Indian"], dietary_tags=["healthy"])
        decision, bundle = self.execute()
        checked = self.verify(decision, bundle, workers=[self.worker()], request={
            **self.request, "cuisine_preferences": ["South Indian"], "dietary_preferences": ["healthy"],
        })
        self.assertEqual(len(self.joined(checked)), 2)

    def test_missing_or_boolean_original_fit_cannot_be_preserved(self):
        decision, bundle = self.execute()
        for value in (None, True):
            with self.subTest(value=value):
                current = copy.deepcopy(decision)
                current["selected_centres"][0]["logistics_fit"] = value
                forged = bundle.model_copy(update={"selections": (
                    bundle.selections[0].model_copy(update={"logistics_fit": value}), bundle.selections[1])})
                checked = self.checked(current, forged)
                self.assertEqual(self.choice(checked, "walk")["logistics_fit"], 0)

    def test_alternative_route_never_repairs_invalid_primary(self):
        decision, bundle = self.execute()
        walk = self.route(decision, "walk")
        walk["alternative_travel"] = copy.deepcopy(walk["travel"])
        walk["alternative_travel"]["total_duration_min"] = 1
        walk["travel"]["total_duration_min"] = None
        self.assertEqual([item["centre_id"] for item in self.checked(decision, bundle)["selected_centres"]], ["fast"])

    def test_captured_hard_limit_cannot_be_relaxed_by_later_request_metadata(self):
        self.fixture.query = self.fixture.query.model_copy(update={"max_travel_time_min": 10})
        decision, bundle = self.execute()
        decision["selected_centres"].append({"centre_id": "walk", "rank": 2, "logistics_fit": 1, "rationale": "Forged"})
        checked = self.checked(decision, bundle, request={**self.request, "max_travel_time_min": 100})
        self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["fast"])

    def test_catalogue_only_search_does_not_gain_ranking_three_selection_cap(self):
        decision, _ = self.execute()
        decision.update(logistics=[], origin=None, selected_centres=[{
            "centre_id": item.centre_id, "rank": rank, "logistics_fit": 1, "rationale": "Catalogue",
        } for rank, item in enumerate(self.fixture.routes, start=1)])
        checked = self.checked(decision, request={**self.request, "search_scope_mode": "global"})
        self.assertEqual(len(checked["selected_centres"]), 4)

    def test_disagreeing_location_envelopes_cannot_union_food_approval(self):
        decision, bundle = self.execute()
        other = copy.deepcopy(decision)
        other["selected_centres"] = [{"centre_id": "direct", "rank": 1, "logistics_fit": 0, "rationale": "Other"}]
        results = verifier.verify_results([
            orchestrator._result_envelope("location", decision),
            orchestrator._result_envelope("location", other), self.worker(),
        ], self.request, location_grounding=bundle, request_id=self.fixture.query.request_id)
        self.assertEqual(self.joined(results), [])

    def test_incoming_candidate_join_cannot_shadow_verifier_owned_approval(self):
        decision, bundle = self.execute()
        forged = {"agent": "candidate_join", "verified": True, "candidates": [self.menu[2]]}
        results = verifier.verify_results([
            forged, orchestrator._result_envelope("location", decision), self.worker(),
        ], self.request, location_grounding=bundle, request_id=self.fixture.query.request_id)
        self.assertEqual(len([item for item in results if item["agent"] == "candidate_join"]), 1)
        self.assertEqual({item["hawker_centre_id"] for item in self.joined(results)}, {"walk", "fast"})

    def test_comparison_only_and_menu_removed_ids_cannot_gain_approval_from_public_metadata(self):
        decision, bundle = self.execute()
        decision["selected_centres"].append({"centre_id": "direct", "rank": 3, "logistics_fit": 1, "rationale": "Forged"})
        results = self.verify(decision, bundle, workers=[self.worker()])
        self.assertEqual({item["hawker_centre_id"] for item in self.joined(results)}, {"walk", "fast"})
        self.fixture.matches["fast"] = []
        state = self.fixture.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        self.assertEqual(handoff_fixtures.LocationGroundingHandoffTests.selection(
            state["location_grounding"], "fast").status_reason, "not_selected")
        state["location_result"]["selected_centres"].append(decision["selected_centres"][1])
        checked = self.checked(state["location_result"], state["location_grounding"])
        self.assertNotIn("fast", {item["centre_id"] for item in checked["selected_centres"]})

    def test_expansion_horizon_is_selection_scope_not_comparison_authority(self):
        self.fixture.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        decision, bundle = self.execute()
        decision["search_expansion"] = {"outcome": "farther_match_requires_approval", "search_horizon_min": 10}
        results = self.verify(decision, bundle, workers=[self.worker()])
        self.assertEqual({item["hawker_centre_id"] for item in self.joined(results)}, {"fast"})
        decision["search_expansion"]["search_horizon_min"] = 15
        checked = self.checked(decision, bundle)
        self.assertEqual(self.choice(checked, "walk")["logistics_fit"], 0.9)

    def test_programming_errors_propagate_instead_of_becoming_proof_rejection(self):
        decision, bundle = self.execute()
        with patch.object(verifier, "_render_verified_choice", side_effect=TypeError("Implementation bug")):
            with self.assertRaisesRegex(TypeError, "Implementation bug"):
                self.checked(decision, bundle)

    def test_actual_graph_carries_private_handoff_across_worker_dispatch_once(self):
        tasks = [PlannedTask(agent="location", task="Find centres"), PlannedTask(agent="dietary", task="Find food")]
        with patch.object(orchestrator, "build_model", return_value={}), \
             patch.object(orchestrator, "parse_request", return_value={"parsed_request": self.request, "tasks": tasks}), \
             patch("agents.dietary_agent.run", return_value=self.worker()), \
             patch.object(orchestrator, "synthesizer", return_value={"final_recommendation": "Offline reply"}):
            graph = orchestrator.build_workflow()
            result = graph.invoke({"request_id": self.fixture.query.request_id, "user_request": self.fixture.query.request,
                                   "tasks": [], "agent_results": []})
        self.assertIsNone(result["location_grounding"])
        location = next(row["payload"] for row in result["verified_results"] if row["agent"] == "location")
        self.assertEqual([item["centre_id"] for item in location["selected_centres"]], ["walk", "fast"])
        self.assertTrue(all(item["logistics_fit"] == 0.9 for item in location["selected_centres"]))
        self.assertEqual({row["hawker_centre_id"] for row in self.joined(result["verified_results"])}, {"walk", "fast"})
        self.fixture.response.assert_called_once()

    def test_no_location_queue_only_and_clarification_paths_keep_existing_boundaries(self):
        self.assertEqual(verifier.verify_results([], {"intent": "queue"}), [])
        self.assertEqual(verifier.verify_results([], {"intent": "other"}), [])
        from test_queue_guardrails import _candidate
        queue = {"agent": "queue", "candidates": [_candidate(
            stall_id=self.menu[0]["stall_id"], centre_id=self.menu[0]["hawker_centre_id"])]}
        with patch.object(verifier, "get_menu_stall_record", return_value=self.menu[0]):
            checked = verifier.verify_results([queue], {"intent": "queue", "max_queue_min": 10})
        self.assertTrue(checked[0]["verified"])
        self.assertEqual(len(checked[0]["candidates"]), 1)
        decision, bundle = self.execute()
        state = self.fixture.state(decision, bundle)
        state["parsed_request"]["clarification_question"] = "Which starting point?"
        self.assertIsNone(orchestrator.clarification_response(state)["location_grounding"])
        state["needs_location"] = True
        self.assertIsNone(orchestrator.location_required_response(state)["location_grounding"])

    def test_farther_approval_summary_does_not_approve_farther_food(self):
        decision, bundle = self.execute()
        self.fixture.matches["walk"] = self.fixture.matches["fast"] = self.fixture.matches["unknown"] = []
        self.route(decision, "direct")["travel"]["total_duration_min"] = 40
        state = self.fixture.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        results = self.verify(state["location_result"], state["location_grounding"], workers=[self.worker()])
        self.assertEqual(results[0]["payload"]["search_expansion"]["outcome"], "farther_match_requires_approval")
        self.assertNotIn("direct", {item["hawker_centre_id"] for item in self.joined(results)})

    def test_private_handoff_is_consumed_and_absent_from_public_api_session_trace_and_logs(self):
        decision, bundle = self.execute()
        state = self.fixture.state(decision, bundle)
        token, events = begin_activity_capture()
        try:
            updates = orchestrator.verify_results_node(state)
        finally:
            end_activity_capture(token)
        self.assertIsNone(updates["location_grounding"])
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        workflow = Mock()
        workflow.invoke.return_value = {**state, **updates, "final_recommendation": "Offline reply"}
        with patch.object(main, "build_workflow", return_value=workflow), \
             patch.object(main, "_resolve_session", side_effect=lambda payload: (None, payload)), \
             patch.object(main, "save_trace") as save_trace:
            response = main.app.test_client().post("/api/chat", json={"message": "Find food nearby"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for data in (updates["verified_results"], body, main._session_state_patch(body), save_trace.call_args.args[0], events):
            text = json.dumps(data)
            self.assertNotIn("location_grounding", text)
            self.assertNotIn(bundle.execution_id, text)
        self.fixture.response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
