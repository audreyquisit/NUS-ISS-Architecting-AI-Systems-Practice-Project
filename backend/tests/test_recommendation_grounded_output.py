"""Actual verifier/conflict/Recommendation/synthesis pipeline with offline model outputs."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents import conflict_resolver, orchestrator, recommendation_agent, recommendation_rendering as rendering, verifier
from models.orchestration import PlannedTask
from services.activity_log import begin_activity_capture, end_activity_capture
from test_location_grounded_reasoning import comparative_reason, grounded_ranking
import test_verifier_location_grounding as fixtures


class GroundedRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.VerifierLocationGroundingTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.fixture.patches.enter_context(patch.object(rendering, "load_hawker_centres", return_value=self.case.fixture.catalogue))
        self.case.fixture.patches.enter_context(patch.object(rendering, "get_menu_item_record", side_effect=lambda key:
            next((row for row in self.case.menu if row["menu_item_id"] == key), None)))

    def state(self, decision=None, bundle=None, *, request=None, workers=None, reused=False):
        if decision is None:
            decision, bundle = self.case.execute()
        request = self.case.request if request is None else request
        public, context = verifier._verify_results_with_context([
            orchestrator._result_envelope("location", decision),
            *(workers if workers is not None else [self.case.worker()])], request,
            location_grounding=bundle, request_id=self.case.fixture.query.request_id, location_reused=reused)
        public = conflict_resolver.resolve_conflicts(public, request)
        return {"request_id": self.case.fixture.query.request_id, "user_request": "Find food nearby",
                "parsed_request": request, "verified_results": public, "recommendation_context": context}

    @staticmethod
    def plan(*ids):
        return {"choices": [{"menu_item_id": key, "reason_refs": ["eligible_menu"]} for key in ids]}

    def synthesize(self, state, output=None, error=None):
        response = Mock(return_value=self.plan("fast:rice", "walk:rice") if output is None else output, side_effect=error)
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(response)
        with patch.object(recommendation_agent, "ChatOpenAI", return_value=model) as construct:
            reply = orchestrator.synthesizer(state)["final_recommendation"]
        return reply, response, model, construct

    def test_valid_plan_keeps_location_order_and_uses_verified_facts_once(self):
        state = self.state()
        reply, response, model, construct = self.synthesize(state)
        self.assertLess(reply.index("1. Fixture walk"), reply.index("2. Fixture fast"))
        self.assertIn("15 min by public transport", reply)
        self.assertIn("Approximate walking time: 1 min", reply)
        self.assertIn("Known transfers: 1", reply)
        self.assertIn("model-reported and uncalibrated", reply)
        self.assertNotIn("food ranking was unavailable", reply)
        response.assert_called_once()
        model.with_structured_output.assert_called_once_with(rendering._RecommendationPlan, include_raw=True)
        self.assertEqual(construct.call_args.kwargs["max_retries"], 0)
        self.case.fixture.response.assert_called_once()
        self.case.fixture.collect.assert_called_once()

    def test_adversarial_prose_and_extra_fields_cannot_become_final_claims(self):
        state = self.state()
        for output in ("Fixture direct: 1 minute, zero walking; fully sheltered and wheelchair accessible",
                       {**self.plan("fast:rice", "walk:rice"), "travel_time": 1, "sheltered": True},
                       AIMessage(content=[{"type": "text", "text": "1 minute wheelchair accessible"}])):
            with self.subTest(output=type(output).__name__):
                reply, response, _, _ = self.synthesize(state, output)
                self.assertIn("deterministic eligible selections", reply)
                self.assertIn("15 min by public transport", reply)
                for unsupported in ("Fixture direct", "wheelchair accessible", "fully sheltered", "zero walking", "route: 1 min"):
                    self.assertNotIn(unsupported, reply)
                response.assert_called_once()

    def test_invalid_duplicate_malformed_and_comparison_menu_ids_fall_back(self):
        state = self.state()
        for output in (self.plan("direct:rice", "walk:rice"), self.plan("walk:rice", "walk:rice"),
                       self.plan("missing"), self.plan([]), self.plan({}), self.plan(12), {"choices": []},
                       {"choices": [{"menu_item_id": "walk:rice", "reason_refs": ["listed_within_budget"]},
                                    {"menu_item_id": "fast:rice"}]},
                       {"choices": [{"menu_item_id": "walk:rice", "reason_refs": ["sheltered"]}]}):
            with self.subTest(output=output):
                reply, response, _, _ = self.synthesize(state, output)
                self.assertIn("deterministic eligible selections", reply)
                self.assertIn("1. Fixture walk", reply)
                self.assertIn("2. Fixture fast", reply)
                self.assertNotIn("Fixture direct", reply)
                response.assert_called_once()

    def test_timeout_and_parser_failures_have_no_repair_call(self):
        state = self.state()
        for output, error in ((None, TimeoutError()), ({"parsed": None, "parsing_error": ValueError("Bad tool")}, None)):
            reply, response, _, _ = self.synthesize(state, output, error)
            self.assertIn("deterministic eligible selections", reply)
            response.assert_called_once()

    def test_programming_errors_still_propagate(self):
        with self.assertRaisesRegex(TypeError, "Implementation bug"):
            self.synthesize(self.state(), error=TypeError("Implementation bug"))

    def test_empty_approval_or_food_join_never_invokes_model(self):
        decision, bundle = self.case.execute()
        for request, reused in (({"intent": "food_discovery", "max_travel_time_min": 10}, True),
                                ({"intent": "food_discovery", "dietary_preferences": ["vegetarian"]}, False)):
            state = self.state(decision, bundle, request=request, reused=reused)
            reply, response, _, construct = self.synthesize(state)
            response.assert_not_called()
            construct.assert_not_called()
            self.assertNotIn("Fixture walk", reply)
        fresh = self.state(decision, bundle, request={"intent": "food_discovery", "max_travel_time_min": 10})
        reply, response, _, _ = self.synthesize(fresh)
        self.assertIn("1. Fixture fast", reply)
        self.assertNotIn("Fixture walk", reply)
        response.assert_called_once()

    def test_only_verifier_accepted_comparisons_can_appear_in_rationales(self):
        self.case.fixture.response.return_value.selected_centres[0].reasons = [comparative_reason(
            "walk", "direct", "LESS_MEASURED_WALKING", ["walking_distance_m"])]
        state = self.state()
        reply, _, _, _ = self.synthesize(state)
        self.assertIn("Fixture direct", reply)  # Accepted comparison evidence, not a recommendation.
        self.assertIn("1. Fixture walk", reply)
        self.assertIn("2. Fixture fast", reply)
        self.assertNotIn("3. Fixture direct", reply)
        rejected = self.plan("walk:rice", "fast:rice")
        rejected["choices"][0]["reason_refs"] = ["UNSUPPORTED_COMPARISON"]
        reply, response, _, _ = self.synthesize(state, rejected)
        self.assertIn("deterministic eligible selections", reply)
        self.assertNotIn("UNSUPPORTED_COMPARISON", reply)
        response.assert_called_once()

    def test_wrong_missing_or_serialized_context_fails_closed(self):
        state = self.state()
        for context in (None, state["recommendation_context"].model_dump(),
                        state["recommendation_context"].model_copy(update={"request_id": "wrong"})):
            reply, response, _, construct = self.synthesize({**state, "recommendation_context": context})
            self.assertIn("No verified eligible food options", reply)
            response.assert_not_called()
            construct.assert_not_called()

    def test_fallback_supplement_and_catalogue_provenance_are_honest(self):
        self.case.fixture.response.side_effect = TimeoutError()
        decision, bundle = self.case.execute()
        reply, _, _, _ = self.synthesize(self.state(decision, bundle), self.plan("fast:rice", "unknown:rice", "direct:rice"))
        self.assertIn("Location deterministic ranking fallback", reply)
        self.assertIn("policy value", reply)
        self.assertIn("Walking distance unknown", reply)
        self.case.fixture.response.side_effect = None
        self.case.fixture.response.return_value = grounded_ranking(["walk", "fast"])
        decision, bundle = self.case.execute()
        self.case.fixture.matches["walk"] = self.case.fixture.matches["fast"] = []
        prior = self.case.fixture.state(decision, bundle)
        prior.update(orchestrator.expand_menu_search(prior))
        reply, _, _, _ = self.synthesize(self.state(prior["location_result"], prior["location_grounding"]), self.plan("unknown:rice", "direct:rice"))
        self.assertIn("Deterministic menu supplement; not ranked", reply)
        reply, _, _, _ = self.synthesize(self.state(decision, bundle, reused=True))
        self.assertIn("Catalogue-only selection", reply)
        self.assertIn("No fresh route evidence", reply)
        self.assertNotIn("Verified primary route:", reply)

    def test_catalogue_or_join_tampering_cannot_reuse_menu_ownership(self):
        state = self.state()
        joined = next(row for row in state["verified_results"] if row["agent"] == "candidate_join")
        for row in joined["candidates"]:
            row["price_label"] = "$0.01 invented"
        reply, response, _, _ = self.synthesize(state)
        self.assertNotIn("$0.01", reply)
        response.assert_not_called()

    def test_balanced_cap_represents_all_approved_centres_before_food_ranking(self):
        self.case.fixture.response.return_value = grounded_ranking(["walk", "fast", "direct"])
        decision, bundle = self.case.execute()
        rows = []
        for centre in ("fast", "walk", "direct"):
            source = next(item for item in self.case.menu if item["hawker_centre_id"] == centre)
            rows.extend({**source, "menu_item_id": centre + ":many:" + str(index)} for index in range(40))
        self.case.menu.extend(rows)
        with patch("services.hawker_service._menu_catalogue", return_value=tuple(self.case.menu)):
            public, context = verifier._verify_results_with_context([
                orchestrator._result_envelope("location", decision), self.case.worker(items=rows)],
                self.case.request, location_grounding=bundle, request_id=self.case.fixture.query.request_id)
        joined = self.case.joined(public)
        self.assertEqual(len(joined), 30)
        self.assertEqual([item["hawker_centre_id"] for item in joined[:3]], ["walk", "fast", "direct"])
        self.assertEqual({centre: sum(item["hawker_centre_id"] == centre for item in joined)
                          for centre in ("walk", "fast", "direct")}, {"walk": 10, "fast": 10, "direct": 10})
        public = conflict_resolver.resolve_conflicts(public, {})
        self.assertEqual([item["hawker_centre_id"] for item in self.case.joined(public)[:10]], ["walk"] * 10)
        reply, response, _, _ = self.synthesize({"request_id": self.case.fixture.query.request_id,
            "user_request": "Find food", "parsed_request": {}, "verified_results": public,
            "recommendation_context": context}, self.plan("direct:many:1", "fast:many:1", "walk:many:1"))
        self.assertLess(reply.index("1. Fixture walk"), reply.index("2. Fixture fast"))
        self.assertIn("3. Fixture direct", reply)
        response.assert_called_once()

    def test_zero_price_queue_and_unknown_values_rank_within_location_centres(self):
        weights = conflict_resolver._weights(["cheap", "short queue"])
        zero = {"price_sgd_min": 0, "price_sgd_max": 8, "queue_minutes": 0}
        positive = {"price_sgd_min": 4, "queue_minutes": 8}
        unknown = {"price_sgd_min": None, "queue_minutes": None}
        with patch.object(conflict_resolver, "_menu_count_by_stall", return_value={}):
            self.assertGreater(conflict_resolver._score(zero, weights), conflict_resolver._score(positive, weights))
            self.assertGreater(conflict_resolver._score(positive, weights), conflict_resolver._score(unknown, weights))
        self.case.menu[1].update(price_sgd_min=0, price_sgd_max=0, price_label="$0.00")
        queue = self.case.queue_worker(("walk", "fast"))
        queue["candidates"][0].update(queue_minutes=0, crowd_level="Low")
        state = self.state(request={"intent": "food_discovery", "budget_amount": 5, "priorities": ["cheap"]},
            workers=[self.case.worker(), queue])
        reply, response, _, _ = self.synthesize(state)
        self.assertIn("1. Fixture walk", reply)
        self.assertIn("$0.00", reply)
        self.assertNotIn("price unavailable", reply)
        self.assertIn("Simulated queue estimate: about 0 min", reply)
        response.assert_called_once()

    def test_model_choices_are_food_ranked_within_a_centre_and_budget_refs_are_validated(self):
        decision, bundle = self.case.execute()
        row = {**self.case.menu[1], "menu_item_id": "walk:free", "menu_item_name": "Free rice",
               "price_sgd_min": 0, "price_sgd_max": 0, "price_label": "$0.00"}
        self.case.menu.append(row)
        with patch("services.hawker_service._menu_catalogue", return_value=tuple(self.case.menu)):
            state = self.state(decision, bundle, request={"intent": "food_discovery", "budget_amount": 5,
                "priorities": ["cheap"]})
        output = {"choices": [
            {"menu_item_id": "walk:rice", "reason_refs": ["starting_price_within_budget"]},
            {"menu_item_id": "fast:rice", "reason_refs": ["starting_price_within_budget"]},
            {"menu_item_id": "walk:free", "reason_refs": ["listed_within_budget"]}]}
        reply, response, _, _ = self.synthesize(state, output)
        self.assertLess(reply.index("Free rice"), reply.index("Chicken rice"))
        self.assertIn("Starting listed price meets the budget; other options may cost more", reply)
        self.assertIn("Listed item prices meet the requested budget", reply)
        self.assertNotIn("food ranking was unavailable", reply)
        response.assert_called_once()

    def test_food_scores_do_not_globally_reverse_location_priority(self):
        decision, bundle = self.case.execute()
        self.case.menu[0].update(price_sgd_min=0, price_sgd_max=0, price_label="$0.00")
        state = self.state(decision, bundle, request={"intent": "food_discovery", "priorities": ["cheap"]})
        self.assertEqual([item["hawker_centre_id"] for item in self.case.joined(state["verified_results"])], ["walk", "fast"])
        reply, _, _, _ = self.synthesize(state)
        self.assertIn("1. Fixture walk", reply)
        self.assertIn("2. Fixture fast", reply)

    def test_food_eligibility_can_remove_a_location_centre_without_reintroducing_it(self):
        decision, bundle = self.case.execute()
        self.case.menu[1]["menu_item_name"] = "Noodles"
        state = self.state(decision, bundle, request={"intent": "food_discovery", "menu_item_preferences": ["chicken rice"]})
        reply, response, _, _ = self.synthesize(state, self.plan("fast:rice"))
        self.assertIn("1. Fixture fast", reply)
        self.assertNotIn("Fixture walk", reply)
        response.assert_called_once()

    def test_directions_use_primary_facts_without_recommendation_model(self):
        state = self.state(request={"intent": "directions"}, workers=[])
        reply, response, _, _ = self.synthesize(state)
        self.assertIn("Verified route information", reply)
        self.assertIn("15 min by public transport", reply)
        self.assertNotIn("Chicken rice", reply)
        response.assert_not_called()

    def test_actual_graph_and_public_projection_do_not_leak_private_context_or_plan(self):
        tasks = [PlannedTask(agent="location", task="Find centres"), PlannedTask(agent="dietary", task="Find food")]
        response = Mock(return_value=self.plan("fast:rice", "walk:rice"))
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(response)
        token, events = begin_activity_capture()
        try:
            with patch.object(orchestrator, "build_model", return_value={}), \
                 patch.object(orchestrator, "parse_request", return_value={"parsed_request": self.case.request, "tasks": tasks}), \
                 patch("agents.dietary_agent.run", return_value=self.case.worker()), \
                 patch.object(recommendation_agent, "ChatOpenAI", return_value=model):
                result = orchestrator.build_workflow().invoke({"request_id": self.case.fixture.query.request_id,
                    "user_request": "Find food nearby", "tasks": [], "agent_results": []})
        finally:
            end_activity_capture(token)
        self.assertIn("1. Fixture walk", result["final_recommendation"])
        context = result["recommendation_context"]
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        graph = Mock()
        graph.invoke.return_value = result
        with patch.object(main, "build_workflow", return_value=graph), \
             patch.object(main, "_resolve_session", side_effect=lambda payload: (None, payload)), \
             patch.object(main, "save_trace") as trace:
            api = main.app.test_client().post("/api/chat", json={"message": "Find food nearby"})
        self.assertEqual(api.status_code, 200)
        body = api.get_json()
        for value in (body, main._session_state_patch(body), trace.call_args.args[0], events):
            text = json.dumps(value)
            for private in (context.context_id, "recommendation_context", "reason_refs", "_RecommendationPlan"):
                self.assertNotIn(private, text)
        prompt = response.call_args.args[0].to_messages()
        self.assertNotIn(context.context_id, str(prompt))
        self.assertNotIn("1.31234567", str(prompt))
        response.assert_called_once()
        self.case.fixture.response.assert_called_once()
        self.case.fixture.collect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
