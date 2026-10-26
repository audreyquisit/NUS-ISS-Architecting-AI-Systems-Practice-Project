"""Offline request-local grounding capture, lifecycle and serialization boundaries."""

import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from agents import location_agent as agent, location_grounding as grounding, orchestrator
from models.location_models import LocationDecision, LocationQuery
from models.orchestration import ParsedRequest, PlannedTask
from services.activity_log import begin_activity_capture, end_activity_capture
from services.location_service import LocationServiceError
from test_location_grounded_reasoning import comparative_reason, grounded_ranking, route


class LocationGroundingHandoffTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch("services.location_service._get", side_effect=AssertionError("No live API")))
        self.patches.enter_context(patch.dict("os.environ", {"LOCATION_DEFAULT_SEARCH_HORIZON_MIN": "30"}))
        self.routes = [route("fast", 5, 1800, 3), route("walk", 15, 100, 1),
                       route("direct", 12, 600, 0), route("unknown", 8)]
        self.catalogue = [{"id": item.centre_id, "name": item.centre_name, "address": item.address,
                           "latitude": item.coordinates.lat, "longitude": item.coordinates.lng}
                          for item in self.routes]
        self.patches.enter_context(patch.object(agent, "load_hawker_centres", return_value=self.catalogue))
        self.resolve = self.patches.enter_context(patch.object(agent, "_resolve_origin", return_value=({
            "lat": 1.31234567, "lng": 103.87654321, "label": "Fixture origin",
        }, "provided_coordinates")))
        self.collect = self.patches.enter_context(patch.object(agent, "_collect_logistics", return_value=self.routes))
        self.response = Mock(return_value=grounded_ranking(["walk", "fast"]))
        self.model = Mock()
        self.model.with_structured_output.return_value = RunnableLambda(self.response)
        self.llm = self.patches.enter_context(patch.object(agent, "_llm", return_value=self.model))
        self.query = LocationQuery(
            request_id="grounding-request", request="Find food nearby",
            travel_mode="pt", parsed_request=ParsedRequest(intent="food_discovery").model_dump(),
        )
        self.matches = {item.centre_id: [{"menu_item_id": item.centre_id + ":rice"}] for item in self.routes}
        self.menu = self.patches.enter_context(patch("services.hawker_service.find_dietary_stalls",
            side_effect=lambda **kw: self.matches.get(kw["candidate_centre_ids"][0], [])))
        self.patches.enter_context(patch("services.hawker_service.find_stalls_within_budget",
            side_effect=AssertionError("No budget requested")))

    def execute(self):
        return agent._run_with_grounding(self.query)

    def state(self, decision, bundle):
        return {
            "request_id": self.query.request_id, "user_request": self.query.request,
            "parsed_request": {**self.query.parsed_request, "menu_item_preferences": ["rice"]},
            "tasks": [PlannedTask(agent="location", task="Find centres")],
            "location_result": decision, "location_grounding": bundle,
            "agent_results": [orchestrator._result_envelope("location", decision)],
        }

    @staticmethod
    def selection(bundle, centre):
        return next(item for item in bundle.selections if item.centre_id == centre)

    def test_grounded_capture_uses_accepted_codes_refs_facts_and_scores(self):
        self.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"],
        )]
        decision, bundle = self.execute()
        self.assertEqual(bundle.version, 1)
        self.assertEqual(bundle.request_id, self.query.request_id)
        self.assertEqual(bundle.decision_source, "grounded_model")
        self.assertEqual(bundle.active_order, ("walk", "fast"))
        choice = self.selection(bundle, "walk")
        self.assertEqual(choice.original_rank, 1)
        self.assertEqual(choice.logistics_fit, decision["selected_centres"][0]["logistics_fit"])
        self.assertEqual(choice.reasons[0].code, "BALANCED_TRADEOFF")
        self.assertIn("direct:primary:walking_distance_m", choice.reasons[0].evidence_refs)
        facts = {fact.id: fact for item in bundle.routes for fact in item.facts}
        self.assertEqual(facts["walk:primary:total_duration_min"].value, 15)
        self.assertEqual(facts["walk:primary:walking_time_min"].provenance, "derived")
        self.assertIsNone(facts["unknown:primary:walking_distance_m"].value)
        self.assertEqual(facts["unknown:primary:walking_distance_m"].provenance, "unavailable")
        self.assertEqual(bundle.confidence, 0.97)
        self.assertEqual(bundle.confidence_basis, "model_reported_uncalibrated")
        self.llm.assert_called_once()
        self.response.assert_called_once()
        self.collect.assert_called_once()

    def test_only_surviving_valid_model_choices_are_captured(self):
        self.response.return_value = grounded_ranking(["unknown_id", "walk", "walk", "fast"])
        _, bundle = self.execute()
        self.assertEqual([item.centre_id for item in bundle.selections], ["walk", "fast"])
        self.assertTrue(all(item.status == "active" for item in bundle.selections))

    def test_fallback_captures_applied_policy_without_parsing_prose(self):
        self.response.side_effect = TimeoutError()
        self.query = self.query.model_copy(update={"priorities": ["least walking"]})
        decision, bundle = self.execute()
        self.assertEqual(bundle.decision_source, "deterministic_fallback")
        self.assertEqual(bundle.fallback_reason, "TimeoutError")
        self.assertEqual(bundle.fallback_order, ("walking_distance_m", "total_duration_min"))
        self.assertEqual(bundle.confidence, 0.5)
        self.assertEqual(bundle.confidence_basis, "deterministic_policy")
        self.assertEqual(bundle.active_order, tuple(item["centre_id"] for item in decision["selected_centres"]))
        self.assertTrue(all(item.source == "deterministic_fallback" and item.logistics_fit == 0.5
                            and item.reasons[0].code == "DETERMINISTIC_ORDER" for item in bundle.selections))
        self.response.assert_called_once()

    def test_public_run_and_standalone_endpoint_return_identical_public_decision(self):
        expected = agent.run(self.query)
        private_decision, bundle = self.execute()
        self.assertEqual(private_decision, expected)
        self.assertEqual(set(expected), set(LocationDecision.model_fields))
        self.assertEqual(self.response.call_count, 2)  # One ranking invocation per execution.
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        response = main.app.test_client().post("/api/location", json=self.query.model_dump(mode="json"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        self.assertEqual(self.response.call_count, 3)
        self.assertNotIn(bundle.execution_id, response.get_data(as_text=True))

    def test_catalogue_and_no_route_paths_do_not_invent_ranking_grounding(self):
        self.query = self.query.model_copy(update={"parsed_request": ParsedRequest(
            intent="food_discovery", search_scope_mode="global",
        ).model_dump()})
        decision, bundle = self.execute()
        self.assertIsNone(bundle)
        self.assertTrue(decision["selected_centres"])
        self.llm.assert_not_called()
        self.query = self.query.model_copy(update={"parsed_request": ParsedRequest(intent="food_discovery").model_dump()})
        self.collect.return_value = []
        decision, bundle = self.execute()
        self.assertEqual(decision["selected_centres"], [])
        self.assertIsNone(bundle)

    def test_unchanged_menu_selection_preserves_matching_grounding(self):
        decision, bundle = self.execute()
        state = self.state(decision, bundle)
        before = copy.deepcopy(state)
        updates = orchestrator.expand_menu_search(state)
        self.assertEqual(state, before)
        self.assertEqual(updates["location_result"]["selected_centres"], decision["selected_centres"])
        self.assertEqual(updates["location_grounding"], bundle)
        self.assertEqual(updates["location_grounding"].execution_id, bundle.execution_id)

    def test_menu_filtered_original_is_superseded_without_losing_comparison_evidence(self):
        self.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "fast", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        decision, bundle = self.execute()
        self.matches["fast"] = []
        updates = orchestrator.expand_menu_search(self.state(decision, bundle))
        current = updates["location_grounding"]
        self.assertEqual(current.active_order, ("walk",))
        self.assertEqual(self.selection(current, "walk").status, "active")
        self.assertEqual(self.selection(current, "fast").status, "superseded")
        self.assertEqual(current.routes, bundle.routes)
        self.assertIn("fast:primary:walking_distance_m", self.selection(current, "walk").reasons[0].evidence_refs)
        self.assertEqual(bundle.active_order, ("walk", "fast"))

    def test_supplements_have_deterministic_provenance_without_model_fit_or_reasons(self):
        decision, bundle = self.execute()
        self.matches["walk"] = self.matches["fast"] = []
        updates = orchestrator.expand_menu_search(self.state(decision, bundle))
        current = updates["location_grounding"]
        self.assertEqual(current.active_order, ("unknown", "direct"))
        self.assertEqual(len(current.supplements), 2)
        self.assertTrue(all(item.status == "superseded" for item in current.selections))
        for supplement in current.supplements:
            self.assertEqual(supplement.selection.source, "menu_expansion")
            self.assertEqual(supplement.selection.fit_basis, "not_location_ranked")
            self.assertIsNone(supplement.selection.logistics_fit)
            self.assertIsNone(supplement.selection.original_rank)
            self.assertEqual(supplement.selection.reasons, ())
            self.assertEqual(supplement.route.source, "onemap")

    def test_conflicting_unusable_or_changed_routes_cannot_retain_original_authority(self):
        for change in ("duplicate", "source", "duration"):
            with self.subTest(change=change):
                decision, bundle = self.execute()
                walk = next(item for item in decision["logistics"] if item["centre_id"] == "walk")
                if change == "duplicate":
                    altered = copy.deepcopy(walk)
                    altered["travel"]["total_duration_min"] = 25
                    decision["logistics"].append(altered)
                elif change == "source":
                    walk["travel"]["source"] = "unavailable"
                else:
                    walk["travel"]["total_duration_min"] = 25
                updates = orchestrator.expand_menu_search(self.state(decision, bundle))
                current = updates["location_grounding"]
                self.assertEqual(self.selection(current, "walk").status, "superseded")
                self.assertEqual(self.selection(current, "walk").status_reason, "route_evidence_changed")
                self.assertNotIn("walk", current.active_order)
                self.assertFalse(any(item.selection.centre_id == "walk" for item in current.supplements))

    def test_changed_unselected_comparison_route_invalidates_dependent_grounding(self):
        self.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        decision, bundle = self.execute()
        direct = next(item for item in decision["logistics"] if item["centre_id"] == "direct")
        direct["travel"]["walking_distance_m"] = 50
        updates = orchestrator.expand_menu_search(self.state(decision, bundle))
        choice = self.selection(updates["location_grounding"], "walk")
        self.assertEqual(choice.status, "superseded")
        self.assertEqual(choice.status_reason, "comparison_evidence_changed")

    def test_integer_float_duplicate_order_preserves_grounding(self):
        decision, bundle = self.execute()
        original = next(item for item in decision["logistics"] if item["centre_id"] == "walk")
        duplicate = copy.deepcopy(original)
        duplicate["travel"]["walking_distance_m"] = 100
        for pair in ((original, duplicate), (duplicate, original)):
            with self.subTest(first_type=type(pair[0]["travel"]["walking_distance_m"]).__name__):
                current = copy.deepcopy(decision)
                current["logistics"] = [item for item in current["logistics"]
                                        if item["centre_id"] != "walk"] + list(pair)
                self.assertEqual(grounding._reconcile_grounding(bundle, current, self.query.request_id), bundle)
                self.assertEqual(orchestrator.expand_menu_search(self.state(current, bundle))["location_grounding"], bundle)
        self.response.assert_called_once()

    def test_signed_zero_duplicate_order_preserves_grounding(self):
        self.routes[1] = route("walk", 15, -0.0, 1)
        decision, bundle = self.execute()
        original = next(item for item in decision["logistics"] if item["centre_id"] == "walk")
        duplicate = route("walk", 15, 0.0, 1).model_dump(mode="json")
        for pair in ((original, duplicate), (duplicate, original)):
            with self.subTest(first_value=repr(pair[0]["travel"]["walking_distance_m"])):
                current = copy.deepcopy(decision)
                current["logistics"] = [item for item in current["logistics"]
                                        if item["centre_id"] != "walk"] + list(pair)
                self.assertEqual(grounding._reconcile_grounding(bundle, current, self.query.request_id), bundle)
                self.assertEqual(orchestrator.expand_menu_search(self.state(current, bundle))["location_grounding"], bundle)
        self.response.assert_called_once()

    def test_genuine_primary_conflicts_remain_rejected_in_both_orders(self):
        decision, bundle = self.execute()
        original = next(item for item in decision["logistics"] if item["centre_id"] == "walk")
        for section, field, value in (("travel", "total_duration_min", 25),
                                      ("travel", "transfers", 2),
                                      ("coordinates", "lat", 1.34),
                                      (None, "route_id", "different-primary-route")):
            duplicate = copy.deepcopy(original)
            (duplicate[section] if section else duplicate)[field] = value
            for pair in ((original, duplicate), (duplicate, original)):
                with self.subTest(field=field, changed_first=pair[0] is duplicate):
                    current = copy.deepcopy(decision)
                    current["logistics"] = [item for item in current["logistics"]
                                            if item["centre_id"] != "walk"] + list(pair)
                    usable, excluded = grounding._usable_primary_routes(current["logistics"])
                    self.assertTrue(excluded)
                    self.assertNotIn("walk", usable)
                    updated = orchestrator.expand_menu_search(self.state(current, bundle))["location_grounding"]
                    self.assertEqual(self.selection(updated, "walk").status_reason, "route_evidence_changed")
                    self.assertNotIn("walk", updated.active_order)
                    self.assertEqual(self.selection(updated, "fast").status, "active")

    def test_boolean_measurements_cannot_alias_numeric_duplicates_or_be_usable(self):
        for value in (False, True):
            base = route("walk", int(value), int(value), int(value)).model_dump(mode="json")
            for metric in ("total_duration_min", "walking_distance_m", "transfers", "walking_time_min"):
                invalid = copy.deepcopy(base)
                invalid["travel"][metric] = value
                for records in ([invalid], [base, invalid], [invalid, base]):
                    with self.subTest(metric=metric, value=value, record_count=len(records),
                                      invalid_first=records[0] is invalid):
                        usable, excluded = grounding._usable_primary_routes(records)
                        self.assertTrue(excluded)
                        self.assertNotIn("walk", usable)

    def test_numeric_canonicalization_preserves_distinct_values_and_json_types(self):
        base = route("walk", 15, 100, 1).model_dump(mode="json")
        for metric, value in (("walking_distance_m", "100"), ("walking_distance_m", None),
                              ("walking_distance_m", 100.00000000000001),
                              ("transfers", True), ("transfers", "1")):
            duplicate = copy.deepcopy(base)
            duplicate["travel"][metric] = value
            for records in ([base, duplicate], [duplicate, base]):
                with self.subTest(metric=metric, value=value, changed_first=records[0] is duplicate):
                    usable, excluded = grounding._usable_primary_routes(records)
                    self.assertTrue(excluded)
                    self.assertNotIn("walk", usable)
        integer = copy.deepcopy(base)
        integer["travel"]["walking_distance_m"] = 2 ** 53 + 1
        rounded_float = copy.deepcopy(integer)
        rounded_float["travel"]["walking_distance_m"] = float(2 ** 53 + 1)
        usable, excluded = grounding._usable_primary_routes([integer, rounded_float])
        self.assertTrue(excluded)
        self.assertNotIn("walk", usable)

    def test_unreferenced_route_changes_preserve_independent_grounding(self):
        self.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        decision, bundle = self.execute()
        unreferenced = next(item for item in decision["logistics"] if item["centre_id"] == "unknown")
        unreferenced["travel"]["total_duration_min"] = 9
        self.assertEqual(grounding._reconcile_grounding(bundle, decision, self.query.request_id), bundle)
        self.assertEqual(orchestrator.expand_menu_search(self.state(decision, bundle))["location_grounding"], bundle)
        self.response.assert_called_once()

    def test_model_metadata_changes_cannot_keep_matching_grounding(self):
        for field, value in (("rationale", "Invented shelter claim"), ("logistics_fit", 1.0)):
            with self.subTest(field=field):
                decision, bundle = self.execute()
                decision["selected_centres"][0][field] = value
                current = grounding._reconcile_grounding(bundle, decision, self.query.request_id)
                self.assertEqual(self.selection(current, "walk").status_reason, "metadata_changed")

    def test_worker_gets_fresh_private_result_once_and_keeps_it_out_of_envelopes(self):
        state = {"request_id": self.query.request_id, "user_request": self.query.request,
                 "parsed_request": self.query.parsed_request,
                 "tasks": [PlannedTask(agent="location", task="Resolve origin")]}
        with patch.object(agent, "_run_with_grounding", wraps=agent._run_with_grounding) as execute, \
             patch.object(agent, "run", side_effect=AssertionError("Do not execute twice")):
            updates = orchestrator.location_worker(state)
        execute.assert_called_once()
        self.response.assert_called_once()
        self.assertIsInstance(updates["location_grounding"], grounding._LocationGrounding)
        self.assertEqual(updates["agent_results"][0]["evidence"], [])
        self.assertNotIn("location_grounding", updates["location_result"])
        self.assertNotIn(updates["location_grounding"].execution_id, json.dumps(updates["agent_results"]))

    def test_location_errors_and_absent_task_clear_fresh_grounding(self):
        decision, bundle = self.execute()
        state = self.state(decision, bundle)
        for error in (agent.MissingOriginError("Missing"), agent.LocationResolutionError("Unknown"),
                      LocationServiceError("Unavailable")):
            with self.subTest(error=type(error).__name__), patch.object(agent, "_run_with_grounding", side_effect=error):
                self.assertIsNone(orchestrator.location_worker(state)["location_grounding"])
        state["tasks"] = []
        self.assertIsNone(orchestrator.location_worker(state)["location_grounding"])

    def test_cache_client_dict_and_clarification_paths_cannot_mint_fresh_grounding(self):
        decision, bundle = self.execute()
        state = self.state(decision, bundle)
        state["conversation_context"] = {"state": {"location_result": decision,
                                                   "location_grounding": bundle.model_dump()}}
        reused = orchestrator.reuse_location(state)
        self.assertIsNone(reused["location_grounding"])
        self.assertIsNone(orchestrator.expand_menu_search({**state, **reused})["location_grounding"])
        state["location_grounding"] = bundle.model_dump()
        self.assertIsNone(orchestrator.expand_menu_search(state)["location_grounding"])
        for response in (orchestrator.location_required_response, orchestrator.clarification_response,
                         orchestrator.respond_without_workers):
            self.assertIsNone(response(state)["location_grounding"])

    def test_request_isolation_and_immutable_snapshots(self):
        first_decision, first = self.execute()
        _, second = self.execute()
        self.assertNotEqual(first.execution_id, second.execution_id)
        self.assertEqual(first.routes, second.routes)
        with self.assertRaises(ValidationError):
            first.confidence = 0.1
        first_decision["logistics"][0]["travel"]["total_duration_min"] = 99
        self.assertEqual(first.routes[0].facts[0].value, 5)
        self.assertIsNone(grounding._reconcile_grounding(first, first_decision, "another-request"))

    def test_bundle_bounds_and_no_free_form_or_origin_context(self):
        self.query = self.query.model_copy(update={"request": "Find food nearby PRIVATE-QUOTE-CANARY"})
        self.response.return_value.preference_interpretation.source_quote = "PRIVATE-QUOTE-CANARY"
        _, bundle = self.execute()
        serialized = bundle.model_dump_json()
        self.assertNotIn("PRIVATE-QUOTE-CANARY", serialized)
        self.assertNotIn("1.31234567", serialized)
        self.assertNotIn("103.87654321", serialized)
        for field, value in (("routes", list(bundle.routes) * 3), ("selections", list(bundle.selections) * 2),
                             ("supplements", [None] * 4), ("active_order", ["x"] * 4),
                             ("prompt", "Do not store prompts")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                grounding._LocationGrounding.model_validate({**bundle.model_dump(), field: value})
        with self.assertRaises(ValidationError):
            grounding._Reason(code="PRIMARY_ROUTE", evidence_refs=["x" * 241])

    def test_eight_original_routes_and_three_supplements_are_bounded(self):
        self.routes[:] = [route(f"r{i:02d}", 5 + i, 100 + i, i % 3) for i in range(20)]
        self.catalogue[:] = [{"id": item.centre_id, "name": item.centre_name, "address": item.address,
                             "latitude": item.coordinates.lat, "longitude": item.coordinates.lng} for item in self.routes]
        self.response.return_value = grounded_ranking(["r00", "r01"])
        decision, bundle = self.execute()
        self.assertEqual(len(bundle.routes), 8)
        self.matches = {name: [{"menu_item_id": name}] for name in ("r10", "r11", "r12", "r13")}
        current = orchestrator.expand_menu_search(self.state(decision, bundle))["location_grounding"]
        self.assertEqual(len(current.routes), 8)
        self.assertEqual(len(current.supplements), 3)
        self.assertEqual(current.active_order, ("r10", "r11", "r12"))

    def test_verifier_boundary_consumes_private_bundle_without_public_leakage(self):
        decision, bundle = self.execute()
        state = self.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        from agents import verifier
        with patch.object(verifier, "load_hawker_centres", return_value=self.catalogue), \
             patch.object(verifier, "verify_results", wraps=verifier.verify_results) as verify:
            updates = orchestrator.verify_results_node(state)
        self.assertEqual(len(verify.call_args.args), 2)
        self.assertEqual(verify.call_args.kwargs, {
            "location_grounding": state["location_grounding"],
            "request_id": self.query.request_id, "location_reused": False,
        })
        self.assertIsNone(updates["location_grounding"])
        verified = updates["verified_results"][0]["payload"]
        self.assertEqual(verified["selected_centres"], decision["selected_centres"])
        self.assertNotIn(bundle.execution_id, json.dumps(verify.call_args.args))
        self.assertNotIn(bundle.execution_id, json.dumps(updates["verified_results"]))

    def test_verifier_boundary_retains_supplements_only_while_their_routes_match(self):
        decision, bundle = self.execute()
        self.matches["walk"] = self.matches["fast"] = []
        state = self.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        from agents import verifier
        with patch.object(verifier, "verify_results", return_value=[]) as verify:
            self.assertIsNone(orchestrator.verify_results_node(state)["location_grounding"])
            current = verify.call_args.kwargs["location_grounding"]
            self.assertEqual(current, state["location_grounding"])
            unknown = next(item for item in state["location_result"]["logistics"]
                           if item["centre_id"] == "unknown")
            unknown["travel"]["total_duration_min"] = 9
            self.assertIsNone(orchestrator.verify_results_node(state)["location_grounding"])
            current = verify.call_args.kwargs["location_grounding"]
        self.assertEqual(current.active_order, ("direct",))
        self.assertEqual([item.selection.centre_id for item in current.supplements], ["direct"])

    def test_verifier_boundary_clears_reuse_missing_origin_and_clarification(self):
        decision, bundle = self.execute()
        from agents import verifier
        for flags in ({"location_reused": True}, {"needs_location": True},
                      {"parsed_request": {"clarification_question": "Which origin?"}}):
            with self.subTest(flags=flags), patch.object(verifier, "verify_results", return_value=[]):
                state = {**self.state(decision, bundle), **flags}
                self.assertIsNone(orchestrator.verify_results_node(state)["location_grounding"])

    def test_private_capture_programming_errors_are_not_model_failures(self):
        with patch.object(agent, "_capture_grounding", side_effect=TypeError("Capture bug")), \
             patch.object(agent, "_deterministic_ranking") as fallback:
            with self.assertRaisesRegex(TypeError, "Capture bug"):
                self.execute()
        fallback.assert_not_called()
        self.response.assert_called_once()

    def test_private_bundle_is_absent_from_api_session_and_raw_trace_response(self):
        decision, bundle = self.execute()
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        graph_result = {**self.state(decision, bundle), "verified_results": [
            orchestrator._result_envelope("location", decision)], "final_recommendation": "Offline reply"}
        workflow = Mock()
        workflow.invoke.return_value = graph_result
        with patch.object(main, "build_workflow", return_value=workflow), \
             patch.object(main, "_resolve_session", side_effect=lambda payload: (None, payload)), \
             patch.object(main, "save_trace") as save_trace:
            response = main.app.test_client().post("/api/chat", json={"message": "Find food nearby"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["reply"], "Offline reply")
        session_projection = main._session_state_patch(body)
        raw_trace = save_trace.call_args.args[0]
        from services.trace_store import _sanitize
        for representation in (body, session_projection, raw_trace, _sanitize(raw_trace)):
            serialized = json.dumps(representation)
            self.assertNotIn("location_grounding", serialized)
            self.assertNotIn(bundle.execution_id, serialized)


if __name__ == "__main__":
    unittest.main()
