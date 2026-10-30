"""Offline Phase 4 decisions, grounding checks and actual SDK parser boundary."""

import ast
import json
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents import location_agent as agent
from models.location_models import CentreLogistics, LocationDecision, LocationQuery, TravelEstimate
from models.orchestration import ParsedRequest
from services.activity_log import begin_activity_capture, end_activity_capture


def fact_ref(centre_id, metric):
    return f"{centre_id}:primary:{metric}"


def grounded_ranking(ids, confidence=0.97):
    """Valid new model fixtures, also used by the earlier regression modules."""
    return agent._GroundedRanking(
        selected_centres=[agent._GroundedChoice(
            centre_id=centre_id, route_id=f"{centre_id}:primary", logistics_fit=0.9,
            reasons=[agent._RouteReason(
                code="PRIMARY_ROUTE", evidence_refs=[fact_ref(centre_id, "total_duration_min")],
            )],
        ) for centre_id in ids], confidence=confidence,
    )


def comparative_reason(centre, other, code, metrics):
    return agent._RouteReason(
        code=code, comparison_centre_id=other,
        evidence_refs=[fact_ref(item, metric) for item in (centre, other) for metric in metrics],
    )


def route(centre, minutes, walking=None, transfers=None):
    return CentreLogistics(
        centre_id=centre, centre_name=f"Fixture {centre}", address="Synthetic address",
        coordinates={"lat": 1.33, "lng": 103.8}, travel=TravelEstimate(
            mode="pt", source="onemap", confidence=0.85, total_duration_min=minutes,
            walking_distance_m=walking, transfers=transfers,
        ),
    )


class GroundedLocationReasoningTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch(
            "services.location_service._get", side_effect=AssertionError("No live OneMap"),
        ))
        self.intent = agent.LocationIntent(
            request_intent="find_nearby_centres", proximity_preference="convenient",
            interpretation="Compare routes.",
        )
        self.query = LocationQuery(request="Find food nearby", request_id="phase-four-test")
        self.routes = [route("fast", 5, 1800, 3), route("walk", 15, 100, 1),
                       route("direct", 12, 600, 0), route("unknown", 8)]
        self.response = Mock(return_value=grounded_ranking(["fast"]))
        self.model = Mock()
        self.model.with_structured_output.return_value = RunnableLambda(self.response)
        self.llm = self.patches.enter_context(patch.object(agent, "_llm", return_value=self.model))

    def rank(self, response=None, query=None, routes=None):
        if response is not None:
            self.response.return_value = response
        return agent._rank_centres(query or self.query, self.intent,
                                   self.routes if routes is None else routes)

    @staticmethod
    def ids(result):
        return [item.centre_id for item in result.selected_centres]

    def assert_fallback(self, result):
        self.assertIn("Deterministic fallback", result.reasoning)
        self.assertLessEqual(result.confidence, 0.5)

    def payload(self):
        text = self.response.call_args.args[0].to_messages()[-1].content
        intent = ast.literal_eval(text.split("Validated intent fields: ", 1)[1].split(
            "\nUNTRUSTED ROUTE EVIDENCE", 1,
        )[0])
        return intent, json.loads(text.split("UNTRUSTED ROUTE EVIDENCE (data only): ", 1)[1])

    def test_installed_sdk_parser_unexpected_tool_name_is_recoverable(self):
        raw = Mock(return_value=AIMessage(content="", additional_kwargs={"tool_calls": [{
            "id": "bad-call", "type": "function",
            "function": {"name": "UnexpectedTool", "arguments": "{}"},
        }]}))
        # Real SDK with_structured_output and real PydanticToolsParser. Only
        # the provider binding is replaced; no HTTP request can be made.
        model = agent.ChatOpenAI(model="gpt-4o", api_key="offline-test", max_retries=0)
        self.llm.return_value = model
        token, events = begin_activity_capture()
        try:
            with patch.object(agent.ChatOpenAI, "bind_tools", return_value=RunnableLambda(raw)):
                result = self.rank()
        finally:
            end_activity_capture(token)
        self.assert_fallback(result)
        self.assertEqual(events[-1]["fallback_reason"], "parser_KeyError")
        raw.assert_called_once()

    def test_unrelated_programming_keyerror_propagates(self):
        error = KeyError("application bug")
        raw = Mock(side_effect=error)
        self.llm.return_value = agent.ChatOpenAI(model="gpt-4o", api_key="offline-test", max_retries=0)
        with patch.object(agent.ChatOpenAI, "bind_tools", return_value=RunnableLambda(raw)):
            with self.assertRaises(KeyError) as raised:
                self.rank()
        self.assertIs(raised.exception, error)
        raw.assert_called_once()

    def test_installed_sdk_success_parses_grounded_tradeoff_with_one_invocation(self):
        response = grounded_ranking(["walk", "fast"])
        response.selected_centres[0].reasons = [comparative_reason(
            "walk", "fast", "BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"],
        )]
        raw = Mock(return_value=AIMessage(content="", additional_kwargs={"tool_calls": [{
            "id": "valid-call", "type": "function",
            "function": {"name": agent._GroundedRanking.__name__, "arguments": response.model_dump_json()},
        }]}))
        self.llm.return_value = agent.ChatOpenAI(model="gpt-4o", api_key="offline-test", max_retries=0)
        with patch.object(agent.ChatOpenAI, "bind_tools", return_value=RunnableLambda(raw)):
            result = self.rank()
        self.assertEqual(self.ids(result), ["walk", "fast"])
        self.assertNotIn("Deterministic fallback", result.reasoning)
        self.assertIn("Trade-off", result.reasoning)
        raw.assert_called_once()

    def test_priority_conflicts_have_one_authoritative_view_and_fallback(self):
        query = self.query.model_copy(update={
            "priorities": ["fastest"], "parsed_request": ParsedRequest(
                intent="food_discovery", priorities=["least walking"],
            ).model_dump(),
        })
        self.intent = self.intent.model_copy(update={
            "proximity_preference": "least walking", "inferred_priorities": ["least walking"],
        })
        self.rank(query=query)
        payload, _ = self.payload()
        self.assertEqual(payload["preference_profile"]["source"], "query")
        self.assertEqual(payload["preference_profile"]["objectives"], ["total_duration_min"])
        self.assertEqual(payload["proximity_preference"], "fastest")
        self.assertNotIn("least walking", payload["inferred_priorities"])
        self.response.side_effect = TimeoutError()
        result = self.rank(query=query)
        self.assertEqual(self.ids(result)[0], "fast")

    def test_profile_is_normalized_once_per_ranking(self):
        with patch.object(agent, "_normalize_preferences", wraps=agent._normalize_preferences) as normalize:
            self.response.side_effect = TimeoutError()
            self.rank()
        normalize.assert_called_once()

    def test_permissive_walking_is_not_a_minimize_objective(self):
        for text in ("I don't mind walking", "Fastest option; walking is fine"):
            with self.subTest(text=text):
                query = self.query.model_copy(update={"request": text})
                profile = agent._normalize_preferences(query, self.intent)
                self.assertFalse(profile.walking_flexible)
                self.assertNotIn("walking_distance_m", profile.objectives)
                response = grounded_ranking(["fast"])
                response.preference_interpretation = agent._PreferenceReading(
                    walking="flexible", source_quote=text,
                )
                result = self.rank(response, query=query)
                self.assertNotIn("Deterministic fallback", result.reasoning)
                self.assertIn("flexible", result.reasoning)

    def test_structured_walking_preference_wins_over_raw_permissive_hint(self):
        query = self.query.model_copy(update={"request": "Walking is fine", "priorities": ["least walking"]})
        profile = agent._normalize_preferences(query, self.intent)
        self.assertEqual(profile.objectives, ["walking_distance_m"])
        self.assertFalse(profile.walking_flexible)

    def test_structured_permissive_walking_cannot_be_reinterpreted_as_minimize(self):
        response = grounded_ranking(["walk"])
        response.preference_interpretation = agent._PreferenceReading(
            walking="minimize", source_quote="I don't mind walking",
        )
        result = self.rank(response, query=self.query.model_copy(update={
            "request": "I don't mind walking", "priorities": ["walking is fine"],
        }))
        self.assert_fallback(result)

    def test_structured_permissive_walking_cannot_be_promoted_to_a_walking_objective(self):
        response = grounded_ranking(["walk"])
        response.preference_interpretation = agent._PreferenceReading(
            focus="walking", walking="flexible", source_quote="I don't mind walking",
        )
        result = self.rank(response, query=self.query.model_copy(update={
            "request": "I don't mind walking", "priorities": ["walking is fine"],
        }))
        self.assert_fallback(result)

    def test_permissive_current_wording_removes_contradictory_legacy_inference(self):
        self.intent = self.intent.model_copy(update={
            "inferred_priorities": ["least walking"], "proximity_preference": "least walking",
        })
        self.rank(query=self.query.model_copy(update={"request": "I don't mind walking"}))
        payload, _ = self.payload()
        self.assertFalse(payload["preference_profile"]["walking_flexible"])
        self.assertEqual(payload["preference_profile"]["objectives"], [])
        self.assertEqual(payload["inferred_priorities"], [])
        self.assertNotEqual(payload["proximity_preference"], "least walking")

    def test_qualified_negated_and_reversed_permission_keeps_walking_tradeoff(self):
        cases = (
            ("Walking is fine only for short distances; avoid long walks.", "avoid long walks"),
            ("Walking is fine, but not too far.", "not too far"),
            ("It isn't true that walking is fine; avoid long walks.", "avoid long walks"),
            ("I don't mind walking? Actually, avoid long walks.", "avoid long walks"),
            ("Avoid long walks.", "Avoid long walks"),
        )
        for source in ("integrated", "standalone", "legacy_inference"):
            for text, quote in cases:
                with self.subTest(source=source, text=text):
                    self.intent = self.intent.model_copy(update={
                        "inferred_priorities": ["least walking"] if source == "legacy_inference" else [],
                        "proximity_preference": "convenient",
                    })
                    query = self.query.model_copy(update={
                        "request": text,
                        "parsed_request": ParsedRequest(intent="food_discovery").model_dump()
                        if source == "integrated" else None,
                    })
                    profile = agent._normalize_preferences(query, self.intent)
                    self.assertFalse(profile.walking_flexible)
                    self.assertEqual(profile.objectives, ["walking_distance_m"]
                                     if source == "legacy_inference" else [])
                    response = grounded_ranking(["walk", "fast"])
                    response.preference_interpretation = agent._PreferenceReading(
                        focus="walking", walking="minimize", source_quote=quote,
                    )
                    response.selected_centres[0].reasons = [comparative_reason(
                        "walk", "fast", "BALANCED_TRADEOFF",
                        ["total_duration_min", "walking_distance_m"],
                    )]
                    calls = self.response.call_count
                    result = self.rank(response, query=query)
                    self.assertEqual(self.ids(result), ["walk", "fast"])
                    self.assertEqual([item.rank for item in result.selected_centres], [1, 2])
                    self.assertEqual(result.confidence, 0.97)
                    self.assertEqual(result.selected_centres[0].logistics_fit, 0.9)
                    self.assertIn("15 min versus 5 min", result.selected_centres[0].rationale)
                    self.assertIn("about 100 m walking", result.selected_centres[0].rationale)
                    self.assertNotIn("Deterministic fallback", result.reasoning)
                    self.assertEqual(self.response.call_count, calls + 1)

    def test_structured_least_walking_remains_authoritative_with_raw_permission(self):
        for source in ("query", "parsed_request"):
            for text in ("I don't mind walking.", "Fastest option; walking is fine."):
                with self.subTest(source=source, text=text):
                    query = self.query.model_copy(update={
                        "request": text,
                        "priorities": ["least walking"] if source == "query" else [],
                        "parsed_request": ParsedRequest(
                            intent="food_discovery", priorities=["least walking"],
                        ).model_dump(),
                    })
                    profile = agent._normalize_preferences(query, self.intent)
                    self.assertEqual(profile.objectives, ["walking_distance_m"])
                    self.assertFalse(profile.walking_flexible)
                    response = grounded_ranking(["walk", "fast"])
                    response.preference_interpretation = agent._PreferenceReading(
                        focus="walking", walking="minimize", source_quote="least walking",
                    )
                    response.selected_centres[0].reasons = [comparative_reason(
                        "walk", "fast", "BALANCED_TRADEOFF",
                        ["total_duration_min", "walking_distance_m"],
                    )]
                    result = self.rank(response, query=query)
                    self.assertEqual(self.ids(result), ["walk", "fast"])
                    self.assertEqual(result.confidence, 0.97)
                    self.assertNotIn("Deterministic fallback", result.reasoning)
                    response.preference_interpretation.walking = "flexible"
                    result = self.rank(response, query=query)
                    self.assert_fallback(result)
                    self.assertEqual(self.ids(result)[0], "walk")
                    self.assertIn("conflicting_walking_preference", result.reasoning)

    def test_legacy_permissive_inference_is_not_authoritative(self):
        self.intent = self.intent.model_copy(update={
            "inferred_priorities": ["walking is fine"],
        })
        query = self.query.model_copy(update={
            "request": "Walking is fine only for short distances; avoid long walks.",
        })
        self.assertFalse(agent._normalize_preferences(query, self.intent).walking_flexible)
        response = grounded_ranking(["walk", "fast"])
        response.preference_interpretation = agent._PreferenceReading(
            focus="walking", walking="minimize", source_quote="avoid long walks",
        )
        response.selected_centres[0].reasons = [comparative_reason(
            "walk", "fast", "BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"],
        )]
        result = self.rank(response, query=query)
        self.assertEqual(self.ids(result), ["walk", "fast"])
        self.assertEqual(result.confidence, 0.97)
        self.assertNotIn("Deterministic fallback", result.reasoning)

    def test_unusable_ranking_after_qualified_permission_still_uses_fallback(self):
        query = self.query.model_copy(update={
            "request": "Walking is fine only for short distances; avoid long walks.",
            "parsed_request": ParsedRequest(intent="food_discovery").model_dump(),
        })
        response = grounded_ranking(["walk"])
        response.preference_interpretation = agent._PreferenceReading(
            focus="walking", walking="minimize", source_quote="avoid long walks",
        )
        response.selected_centres[0].reasons[0].evidence_refs = ["invented:fact"]
        result = self.rank(response, query=query)
        self.assert_fallback(result)
        self.assertEqual(self.ids(result), ["fast", "unknown", "direct"])
        self.assertEqual(result.confidence, 0.5)
        self.assertIn("no_valid_model_selections", result.reasoning)
        self.response.assert_called_once()
        self.llm.assert_called_once()

    def test_private_output_and_preference_payload_are_bounded(self):
        query = self.query.model_copy(update={"priorities": ["x" * 1000] * 20})
        self.rank(query=query)
        payload, _ = self.payload()
        self.assertEqual(payload["preference_profile"]["objectives"], [])
        self.assertLess(len(json.dumps(payload)), 2500)
        bad = grounded_ranking(["fast"]).model_dump()
        bad["selected_centres"][0]["reasons"][0]["evidence_refs"] = ["x" * 241]
        self.assert_fallback(self.rank(bad))

    def test_justified_nonfastest_tradeoff_retains_model_order(self):
        response = grounded_ranking(["walk", "fast"])
        response.preference_interpretation = agent._PreferenceReading(
            focus="walking", walking="minimize", source_quote="less walking",
        )
        response.selected_centres[0].reasons = [comparative_reason(
            "walk", "fast", "BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"],
        )]
        result = self.rank(response, query=self.query.model_copy(update={"request": "Some extra time for less walking"}))
        self.assertEqual(self.ids(result), ["walk", "fast"])
        self.assertEqual(result.confidence, 0.97)
        self.assertIn("15 min versus 5 min", result.selected_centres[0].rationale)
        self.assertIn("Trade-off", result.reasoning)
        self.assertNotIn("Deterministic fallback", result.reasoning)

    def test_false_zero_walking_or_transfer_comparisons_are_rejected(self):
        for code, other, metric in (
            ("LESS_MEASURED_WALKING", "walk", "walking_distance_m"),
            ("FEWER_KNOWN_TRANSFERS", "direct", "transfers"),
        ):
            with self.subTest(code=code):
                response = grounded_ranking(["fast"])
                response.selected_centres[0].reasons = [comparative_reason("fast", other, code, [metric])]
                self.assert_fallback(self.rank(response))

    def test_unsupported_free_prose_is_not_accepted_even_with_valid_ids(self):
        for text in ("wheelchair accessible", "covered shelter", "safe in heavy rain", "exactly 3 minutes walking"):
            with self.subTest(text=text):
                response = grounded_ranking(["fast"]).model_dump()
                response["selected_centres"][0]["rationale"] = text
                result = self.rank(response)
                self.assert_fallback(result)
                self.assertNotIn(text, result.reasoning)

    def test_raw_preference_quote_is_not_rendered_as_a_route_fact(self):
        query = self.query.model_copy(update={"request": "I want shelter from rain"})
        response = grounded_ranking(["fast"])
        response.preference_interpretation.source_quote = query.request
        result = self.rank(response, query=query)
        self.assertNotIn("shelter", result.reasoning)
        self.assertNotIn("rain", result.reasoning)

    def test_unknown_is_distinct_from_measured_zero_in_evidence(self):
        self.rank(routes=[route("fast", 5, 0, 0), route("unknown", 8)])
        _, evidence = self.payload()
        facts = {fact["id"]: fact for record in evidence for fact in record["facts"]}
        for metric in ("walking_distance_m", "transfers"):
            self.assertEqual(facts[fact_ref("fast", metric)]["value"], 0)
            self.assertEqual(facts[fact_ref("fast", metric)]["provenance"], "measured")
            self.assertIsNone(facts[fact_ref("unknown", metric)]["value"])
            self.assertEqual(facts[fact_ref("unknown", metric)]["provenance"], "unavailable")
        self.assertEqual(facts[fact_ref("fast", "walking_time_min")]["provenance"], "derived")

    def test_missing_metric_reason_is_valid_only_for_missing_facts(self):
        response = grounded_ranking(["unknown"])
        response.selected_centres[0].reasons = [agent._RouteReason(
            code="METRIC_UNAVAILABLE", evidence_refs=[fact_ref("unknown", "transfers")],
        )]
        result = self.rank(response)
        self.assertIn("Transfer count unavailable", result.reasoning)
        response = grounded_ranking(["direct"])
        response.selected_centres[0].reasons = [agent._RouteReason(
            code="METRIC_UNAVAILABLE", evidence_refs=[fact_ref("direct", "transfers")],
        )]
        self.assert_fallback(self.rank(response))

    def test_unavailable_or_derived_facts_cannot_support_measured_comparisons(self):
        response = grounded_ranking(["unknown"])
        response.selected_centres[0].reasons = [comparative_reason(
            "unknown", "fast", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        self.assert_fallback(self.rank(response))
        response.selected_centres[0].reasons[0].evidence_refs = [
            fact_ref("unknown", "walking_time_min"), fact_ref("fast", "walking_time_min"),
        ]
        self.assert_fallback(self.rank(response))

    def test_invalid_references_and_wrong_route_identity_trigger_fallback(self):
        for change in ("ref", "route", "other_centre"):
            with self.subTest(change=change):
                response = grounded_ranking(["fast"])
                if change == "ref":
                    response.selected_centres[0].reasons[0].evidence_refs = ["invented:fact"]
                elif change == "route":
                    response.selected_centres[0].route_id = "fast:alternative"
                else:
                    response.selected_centres[0].reasons[0].evidence_refs = [fact_ref("walk", "total_duration_min")]
                self.assert_fallback(self.rank(response))

    def test_alternative_metrics_are_not_model_evidence_or_selectable(self):
        row = self.routes[0].model_copy(update={"alternative_travel": TravelEstimate(
            total_duration_min=45, walking_distance_m=0, transfers=0,
            mode="walk", source="onemap", confidence=0.85,
        )})
        query = self.query.model_copy(update={"max_travel_time_min": 10})
        result = self.rank(query=query, routes=[row])
        self.assertEqual(self.ids(result), ["fast"])
        _, evidence = self.payload()
        self.assertNotIn("alternative", json.dumps(evidence))
        response = grounded_ranking(["fast"])
        response.selected_centres[0].reasons[0].evidence_refs = ["fast:alternative:total_duration_min"]
        self.assert_fallback(self.rank(response, query=query, routes=[row]))

    def test_partial_rejection_removes_dependent_stale_explanations(self):
        response = grounded_ranking(["walk", "fast", "direct"])
        response.selected_centres[0].reasons = [comparative_reason(
            "walk", "fast", "LESS_MEASURED_WALKING", ["walking_distance_m"],
        )]
        response.selected_centres[1].reasons = [comparative_reason(
            "fast", "direct", "FEWER_KNOWN_TRANSFERS", ["transfers"],
        )]
        result = self.rank(response)
        self.assertEqual(self.ids(result), ["direct"])
        self.assertNotIn("Fixture fast", result.reasoning)
        self.assertNotIn("Fixture walk", result.reasoning)
        self.assertTrue(any("stale" in item for item in result.limitations))

    def test_comparison_scope_is_named_candidates_not_a_global_best_claim(self):
        response = grounded_ranking(["direct"])
        response.selected_centres[0].reasons = [comparative_reason(
            "direct", "walk", "SHORTER_DURATION", ["total_duration_min"],
        )]
        result = self.rank(response)
        self.assertIn("12 min versus Fixture walk's 15 min", result.reasoning)
        self.assertNotIn("fastest", result.reasoning)
        response.selected_centres[0].reasons[0].comparison_centre_id = "not_shortlisted"
        self.assert_fallback(self.rank(response))

    def test_eight_candidate_limit_is_explicit_and_model_order_is_kept(self):
        rows = [route(f"r{i:02d}", 5+i) for i in range(20)]
        result = self.rank(grounded_ranking(["r03", "r00", "r01"]), routes=rows)
        self.assertEqual(self.ids(result), ["r03", "r00", "r01"])
        self.assertTrue(any("eight" in item for item in result.limitations))
        _, evidence = self.payload()
        self.assertEqual(len(evidence), 8)

    def test_hard_constraints_and_confidence_bounds_are_not_bypassed(self):
        query = self.query.model_copy(update={"max_travel_time_min": 8})
        result = self.rank(grounded_ranking(["walk"]), query=query)
        self.assert_fallback(result)
        self.assertTrue(set(self.ids(result)) <= {"fast", "unknown"})
        response = grounded_ranking(["fast"]).model_copy(update={"confidence": 1.1})
        self.assert_fallback(self.rank(response))

    def test_fallback_is_transparent_after_unusable_grounded_output(self):
        response = grounded_ranking(["fast"])
        response.selected_centres[0].reasons[0].evidence_refs = ["bad"]
        token, events = begin_activity_capture()
        try:
            result = self.rank(response)
        finally:
            end_activity_capture(token)
        self.assert_fallback(result)
        self.assertEqual(events[-1]["request_id"], self.query.request_id)
        self.assertEqual(events[-1]["ranking_source"], "deterministic_fallback")
        self.assertEqual(events[-1]["fallback_reason"], "no_valid_model_selections")

    def test_normal_success_has_one_model_call_and_public_ranking_fields_only(self):
        result = self.rank()
        self.llm.assert_called_once()
        self.response.assert_called_once()
        self.model.with_structured_output.assert_called_once_with(
            agent._GroundedRanking, method="function_calling", include_raw=True,
        )
        self.assertEqual(set(result.selected_centres[0].model_dump()), {
            "centre_id", "rank", "logistics_fit", "rationale",
        })
        self.assertEqual(set(LocationDecision.model_fields), {
            "selected_centres", "logistics", "reasoning", "confidence", "limitations",
            "origin", "origin_label", "origin_source", "intent_interpretation",
        })


if __name__ == "__main__":
    unittest.main()
