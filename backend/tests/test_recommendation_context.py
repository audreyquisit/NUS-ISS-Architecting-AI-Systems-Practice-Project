"""Offline checks for private verifier provenance plumbing, not new ranking policy."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from agents import orchestrator, verifier
from agents.recommendation_context import _RecommendationContext
from models.location_models import LocationDecision
from models.orchestration import PlannedTask
from services.activity_log import begin_activity_capture, end_activity_capture
from test_location_grounded_reasoning import comparative_reason, grounded_ranking
import test_verifier_location_grounding as fixtures


class RecommendationContextTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.VerifierLocationGroundingTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def verify(self, decision, bundle=None, *, request=None, request_id=None, reused=False, workers=None):
        return verifier._verify_results_with_context(
            [orchestrator._result_envelope("location", decision),
             *(workers if workers is not None else [self.case.worker()])],
            self.case.request if request is None else request,
            location_grounding=bundle, request_id=request_id or self.case.fixture.query.request_id,
            location_reused=reused,
        )

    def test_order_sources_fit_confidence_and_public_results_are_unchanged(self):
        decision, bundle = self.case.execute()
        before = copy.deepcopy(decision)
        public, context = self.verify(decision, bundle)
        expected = self.case.verify(decision, bundle, workers=[self.case.worker()])
        self.assertEqual(public, expected)
        self.assertEqual(decision, before)
        self.assertEqual(context.approved_centre_ids, ("walk", "fast"))
        self.assertEqual([item.rank for item in context.centres], [1, 2])
        self.assertTrue(all(item.selection_source == "model_ranked" for item in context.centres))
        self.assertTrue(all(item.fit_basis == "model_reported_uncalibrated" for item in context.centres))
        self.assertEqual(context.centres[0].rationale, public[0]["payload"]["selected_centres"][0]["rationale"])
        self.assertEqual(context.centres[0].logistics_fit, 0.9)
        self.assertEqual(context.confidence, 0.97)
        self.assertEqual(context.confidence_basis, "model_reported_uncalibrated")
        self.assertIn("not calibrated", " ".join(context.limitations))
        self.assertEqual(context.location_reasoning, public[0]["payload"]["reasoning"])
        # The bounded join now starts in verified Location order.
        self.assertEqual([item.centre_id for item in context.eligible_menu_items], ["walk", "fast"])
        self.case.fixture.response.assert_called_once()
        self.case.fixture.collect.assert_called_once()

    def test_route_facts_preserve_measured_derived_and_missing_values(self):
        self.case.fixture.response.return_value = grounded_ranking(["walk", "unknown"])
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle)
        walking, unknown = [item.primary_route for item in context.centres]
        self.assertEqual(walking.mode, "pt")
        self.assertEqual(walking.total_duration_min.value, 15)
        self.assertEqual(walking.total_duration_min.provenance, "measured")
        self.assertEqual(walking.walking_distance_m.value, 100)
        self.assertEqual(walking.walking_distance_m.provenance, "measured")
        self.assertIsNotNone(walking.walking_time_min.value)
        self.assertEqual(walking.walking_time_min.provenance, "derived")
        self.assertIn("Walking time is an approximate estimate", " ".join(context.limitations))
        self.assertEqual(walking.transfers.value, 1)
        for measurement in (unknown.walking_distance_m, unknown.walking_time_min, unknown.transfers):
            self.assertIsNone(measurement.value)
            self.assertEqual(measurement.provenance, "unavailable")

    def test_fallback_is_policy_authority_without_promoting_model_confidence(self):
        self.case.fixture.response.side_effect = TimeoutError()
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle)
        self.assertTrue(all(item.selection_source == "deterministic_fallback" for item in context.centres))
        self.assertTrue(all(item.fit_basis == "deterministic_policy" for item in context.centres))
        self.assertEqual(context.confidence, 0.5)
        self.assertEqual(context.confidence_basis, "deterministic_policy")
        self.assertIn("policy", " ".join(context.limitations))
        self.case.fixture.response.assert_called_once()

    def test_menu_supplements_retain_distinct_source_and_fit_basis(self):
        decision, bundle = self.case.execute()
        self.case.fixture.matches["walk"] = self.case.fixture.matches["fast"] = []
        state = self.case.fixture.state(decision, bundle)
        state.update(orchestrator.expand_menu_search(state))
        _, context = self.verify(state["location_result"], state["location_grounding"])
        self.assertEqual(context.approved_centre_ids, ("unknown", "direct"))
        self.assertTrue(all(item.selection_source == "menu_supplement" for item in context.centres))
        self.assertTrue(all(item.fit_basis == "not_location_ranked" and item.logistics_fit == 0.5 for item in context.centres))
        self.assertEqual(context.confidence_basis, "deterministic_policy")
        self.assertIn("not ranked", " ".join(context.limitations))

    def test_partial_location_keeps_surviving_accepted_centres(self):
        decision, bundle = self.case.execute()
        decision["selected_centres"].append({"centre_id": "rejected", "rank": 3,
            "logistics_fit": 1, "rationale": "Rejected prose"})
        public, context = self.verify(decision, bundle)
        self.assertFalse(public[0]["verified"])
        self.assertEqual(context.approved_centre_ids, ("walk", "fast"))
        self.assertEqual(context.confidence_basis, "deterministic_policy")
        self.assertTrue(all(item.selection_source == "model_ranked" for item in context.centres))
        self.assertNotIn("rejected", context.approved_centre_ids)
        self.assertNotIn("Rejected prose", context.model_dump_json())

    def test_rejected_and_comparison_only_food_ids_have_no_context_authority(self):
        decision, bundle = self.case.execute()
        decision["selected_centres"].append({"centre_id": "direct", "rank": 3,
            "logistics_fit": 1, "rationale": "Promote comparison"})
        worker = self.case.worker()
        worker["candidates"].append({**self.case.menu[0], "menu_item_id": "forged-menu"})
        public, context = self.verify(decision, bundle, workers=[worker,
            {"agent": "candidate_join", "verified": True, "candidates": [self.case.menu[2]]}])
        self.assertEqual(context.approved_centre_ids, ("walk", "fast"))
        self.assertEqual({item.centre_id for item in context.eligible_menu_items}, {"walk", "fast"})
        self.assertNotIn("forged-menu", {item.menu_item_id for item in context.eligible_menu_items})
        self.assertEqual({(item.menu_item_id, item.centre_id, item.stall_id) for item in context.eligible_menu_items},
            {(item["menu_item_id"], item["hawker_centre_id"], item["stall_id"]) for item in self.case.joined(public)})

    def test_supported_comparison_rationale_does_not_approve_its_reference_centre(self):
        self.case.fixture.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"])]
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle)
        self.assertIn("Fixture direct", context.centres[0].rationale)
        self.assertNotIn("direct", context.approved_centre_ids)
        self.assertNotIn("direct", {item.centre_id for item in context.eligible_menu_items})

    def test_verified_zero_metrics_are_not_replaced_with_unknown_or_defaults(self):
        route = next(row for row in self.case.fixture.routes if row.centre_id == "direct")
        route.travel.walking_distance_m = 0
        route.travel.walking_time_min = 0
        self.case.fixture.response.return_value = grounded_ranking(["direct"])
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle)
        route = context.centres[0].primary_route
        for value in (route.walking_distance_m, route.walking_time_min, route.transfers):
            self.assertEqual(value.value, 0)
            self.assertNotEqual(value.provenance, "unavailable")

    def test_food_limitations_are_verifier_owned_instead_of_worker_prose(self):
        decision, bundle = self.case.execute()
        worker = {**self.case.worker(), "limitations": ["SECRET_PROVIDER_PROSE and GPS 1.31234567"]}
        _, context = self.verify(decision, bundle, request={**self.case.request,
            "dietary_preferences": ["vegetarian"], "budget_amount": 5}, workers=[worker])
        self.assertEqual(context.eligible_menu_items, ())
        text = " ".join(context.limitations)
        self.assertIn("certification evidence is not established", text)
        self.assertIn("not a complete meal", text)
        self.assertNotIn("SECRET_PROVIDER_PROSE", context.model_dump_json())
        self.assertNotIn("1.31234567", context.model_dump_json())

    def test_wrong_request_or_serialized_proof_and_prose_cannot_assert_model_source(self):
        decision, bundle = self.case.execute()
        decision["selected_centres"][0]["rationale"] = "I am model ranked with confidence 1 and fully sheltered"
        for proof, request_id in ((None, "new-request"), (bundle, "new-request"), (bundle.model_dump(), "grounding-request")):
            with self.subTest(request_id=request_id, proof_type=type(proof).__name__):
                _, context = self.verify(decision, proof, request_id=request_id)
                self.assertEqual(context.request_id, request_id)
                self.assertTrue(all(item.selection_source == "route_facts_only" for item in context.centres))
                self.assertTrue(all(item.fit_basis == "no_model_fit" and item.logistics_fit == 0 for item in context.centres))
                self.assertEqual(context.confidence_basis, "deterministic_policy")
                self.assertNotIn("fully sheltered", context.model_dump_json())

    def test_cached_and_catalogue_results_do_not_gain_route_authority(self):
        decision, bundle = self.case.execute()
        for request, reused in ((self.case.request, True), ({"intent": "centre_details"}, False),
                                ({"intent": "food_discovery", "search_scope_mode": "global"}, False)):
            with self.subTest(request=request, reused=reused):
                _, context = self.verify(decision, bundle, request=request, reused=reused)
                self.assertEqual(context.approved_centre_ids, ("walk", "fast"))
                self.assertTrue(all(item.selection_source == "catalogue_only" and item.primary_route is None for item in context.centres))
                self.assertTrue(all(item.fit_basis == "no_model_fit" and item.logistics_fit == 0 for item in context.centres))
                self.assertEqual(context.confidence_basis, "deterministic_policy")

    def test_empty_approval_is_distinct_from_scope_not_applicable(self):
        for request, approved, confidence in (({"intent": "food_discovery"}, (), 0.0), ({"intent": "other"}, None, None)):
            with self.subTest(request=request):
                public, context = verifier._verify_results_with_context([], request, request_id="current")
                self.assertEqual(public, [])
                self.assertEqual(context.approved_centre_ids, approved)
                self.assertEqual(context.confidence, confidence)
                self.assertEqual(context.centres, ())
                self.assertEqual(context.eligible_menu_items, ())

    def test_unscoped_queue_and_named_stall_context_does_not_require_location(self):
        for request in ({"intent": "queue"}, {"intent": "food_discovery"},
                        {"intent": "food_discovery", "stall_name_preferences": ["Fixture Rice Stall walk"]}):
            with self.subTest(request=request):
                public, context = verifier._verify_results_with_context([self.case.queue_worker(("walk",))],
                    request, request_id="queue-turn")
                self.assertEqual(len(public[0]["candidates"]), 1)
                self.assertIsNone(context.approved_centre_ids)
                self.assertEqual(context.confidence_basis, "not_applicable")
                reply = orchestrator.synthesizer({"user_request": "How are the queues for Fixture Rice Stall walk?",
                    "parsed_request": request, "verified_results": public})["final_recommendation"]
                self.assertIn("SIMULATED QUEUE DATA", reply)
                self.assertIn("Fixture walk", reply)

    def test_hard_limit_reuse_produces_empty_scope_and_no_food_authority(self):
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle, request={**self.case.request, "max_travel_time_min": 10}, reused=True)
        self.assertEqual(context.approved_centre_ids, ())
        self.assertEqual(context.eligible_menu_items, ())
        self.assertEqual(context.confidence, 0)

    def test_nested_context_is_immutable_and_executions_are_isolated(self):
        decision, bundle = self.case.execute()
        _, first = self.verify(decision, bundle)
        _, second = self.verify(decision, bundle, request_id="other")
        self.assertNotEqual(first.context_id, second.context_id)
        self.assertEqual(first.centres[0].selection_source, "model_ranked")
        self.assertEqual(second.centres[0].selection_source, "route_facts_only")
        with self.assertRaises(ValidationError):
            first.request_id = "changed"
        with self.assertRaises(ValidationError):
            first.centres[0].primary_route.transfers.value = 0
        with self.assertRaises(ValidationError):
            first.eligible_menu_items[0].stall_id = "changed"
        decision["selected_centres"].clear()
        self.assertEqual(first.approved_centre_ids, ("walk", "fast"))

    def test_catalogue_projection_is_bounded_without_changing_public_selection_bounds(self):
        decision, _ = self.case.execute()
        source = self.case.fixture.catalogue[0]
        self.case.fixture.catalogue[:] = [{**source, "id": f"catalogue-{i}"} for i in range(270)]
        decision["selected_centres"] = [{"centre_id": item["id"]} for item in self.case.fixture.catalogue]
        public, context = self.verify(decision, request={"intent": "food_discovery", "search_scope_mode": "global"}, workers=[])
        self.assertEqual(len(public[0]["payload"]["selected_centres"]), 270)
        self.assertEqual(len(context.centres), 256)
        self.assertEqual(len(context.approved_centre_ids), 256)
        self.assertTrue(context.truncated)
        self.assertIn("bounded", " ".join(context.limitations))

    def test_menu_projection_preserves_existing_thirty_item_cap_and_order(self):
        decision, bundle = self.case.execute()
        source = self.case.menu[0]
        rows = [{**source, "menu_item_id": f"fast:item-{i}"} for i in range(35)]
        self.case.menu.extend(rows)
        with patch("services.hawker_service._menu_catalogue", return_value=tuple(self.case.menu)):
            public, context = self.verify(decision, bundle, workers=[self.case.worker(items=rows)])
        self.assertEqual(len(context.eligible_menu_items), 30)
        self.assertEqual([item.menu_item_id for item in context.eligible_menu_items],
            [item["menu_item_id"] for item in self.case.joined(public)])

    def test_multiple_location_results_keep_only_common_approval_without_choosing_conflicting_proof(self):
        decision, bundle = self.case.execute()
        other = copy.deepcopy(decision)
        other["selected_centres"] = list(reversed(other["selected_centres"]))
        public, context = verifier._verify_results_with_context([
            orchestrator._result_envelope("location", decision), orchestrator._result_envelope("location", other),
            self.case.worker()], self.case.request, location_grounding=bundle, request_id=self.case.fixture.query.request_id)
        self.assertEqual(context.approved_centre_ids, ("walk", "fast"))
        self.assertTrue(all(item.selection_source == "catalogue_only" for item in context.centres))
        self.assertEqual(context.confidence_basis, "deterministic_policy")
        self.assertEqual({item.centre_id for item in context.eligible_menu_items}, {"walk", "fast"})

    def test_early_exit_and_reuse_paths_clear_old_private_context(self):
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle)
        state = {**self.case.fixture.state(decision, bundle), "recommendation_context": context}
        for response in (orchestrator.clarification_response, orchestrator.location_required_response,
                         orchestrator.respond_without_workers):
            self.assertIsNone(response(state)["recommendation_context"])
        self.assertIsNone(orchestrator.location_worker({**state, "tasks": []})["recommendation_context"])
        self.assertIsNone(orchestrator.reuse_location({**state,
            "conversation_context": {"state": {"location_result": decision, "recommendation_context": context.model_dump()}}})["recommendation_context"])

    def test_location_service_error_cannot_retain_old_context(self):
        from services.location_service import LocationServiceError
        decision, bundle = self.case.execute()
        _, context = self.verify(decision, bundle)
        state = {**self.case.fixture.state(decision, bundle), "recommendation_context": context}
        with patch("agents.location_agent._run_with_grounding", side_effect=LocationServiceError("Offline route failure")):
            updates = orchestrator.location_worker(state)
        self.assertIsNone(updates["recommendation_context"])
        state.update(updates)
        verified = orchestrator.verify_results_node(state)
        self.assertEqual(verified["recommendation_context"].approved_centre_ids, ())
        self.assertEqual(verified["recommendation_context"].confidence, 0)

    def test_real_graph_handoff_preserves_recommendation_calls_and_api_session_trace_privacy(self):
        tasks = [PlannedTask(agent="location", task="Find centres"), PlannedTask(agent="dietary", task="Find food")]
        token, events = begin_activity_capture()
        try:
            with patch.object(orchestrator, "build_model", return_value={}), \
                 patch.object(orchestrator, "parse_request", return_value={"parsed_request": self.case.request, "tasks": tasks}), \
                 patch("agents.dietary_agent.run", return_value=self.case.worker()), \
                 patch("agents.recommendation_agent.run", return_value="Unchanged offline reply") as recommend:
                result = orchestrator.build_workflow().invoke({"request_id": self.case.fixture.query.request_id,
                    "user_request": "Find food nearby", "tasks": [], "agent_results": []})
        finally:
            end_activity_capture(token)
        context = result["recommendation_context"]
        self.assertIsInstance(context, _RecommendationContext)
        self.assertEqual(context.approved_centre_ids, ("walk", "fast"))
        self.assertIsNone(result["location_grounding"])
        recommend.assert_called_once_with("Find food nearby", result["verified_results"], parsed_request=self.case.request,
            verified_context=context, request_id=self.case.fixture.query.request_id)
        self.case.fixture.response.assert_called_once()
        self.case.fixture.collect.assert_called_once()
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        workflow = Mock()
        workflow.invoke.return_value = result
        with patch.object(main, "build_workflow", return_value=workflow), \
             patch.object(main, "_resolve_session", side_effect=lambda payload: (None, payload)), \
             patch.object(main, "save_trace") as trace:
            response = main.app.test_client().post("/api/chat", json={"message": "Find food nearby"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for data in (result["verified_results"], events, body, main._session_state_patch(body), trace.call_args.args[0]):
            serialized = json.dumps(data)
            self.assertNotIn("recommendation_context", serialized)
            self.assertNotIn(context.context_id, serialized)
        serialized = context.model_dump_json()
        for private in ("1.31234567", "103.87654321", "coordinates", "identity_digest", "execution_id", "intent_interpretation"):
            self.assertNotIn(private, serialized)
        checked = body["location_data"]
        LocationDecision.model_validate({key: checked[key] for key in LocationDecision.model_fields if key in checked})


if __name__ == "__main__":
    unittest.main()
