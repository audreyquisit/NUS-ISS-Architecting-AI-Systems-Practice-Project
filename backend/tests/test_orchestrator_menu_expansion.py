"""Offline menu-expansion integration; no models, routing or report hooks."""

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from agents import orchestrator
from models.location_models import LocationDecision
from models.orchestration import ParsedRequest, PlannedTask
from services.activity_log import begin_activity_capture, end_activity_capture
from services.hawker_service import find_dietary_stalls


def route(centre, minutes, walking=600, transfers=0):
    return {
        "centre_id": centre, "centre_name": "Fixture " + centre,
        "address": "Fixture address", "coordinates": {"lat": 1.33, "lng": 103.8},
        "travel": {
            "total_duration_min": minutes, "walking_distance_m": walking,
            "transfers": transfers, "mode": "pt", "source": "onemap", "confidence": 0.8,
        },
    }


def selection(centre, rank):
    return {
        "centre_id": centre, "rank": rank, "logistics_fit": 0.91 - rank / 100,
        "rationale": "Grounded Location rationale for " + centre,
    }


class OrchestratorMenuExpansionTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch.dict("os.environ", {"LOCATION_DEFAULT_SEARCH_HORIZON_MIN": "30"}))
        for target in (
            "agents.orchestrator.ChatOpenAI", "services.location_service._get",
            "services.location_service.get_route",
        ):
            self.patches.enter_context(patch(target, side_effect=AssertionError("No live calls")))
        self.routes = [route("direct", 12, 900, 0), route("fast", 5, 100, 3),
                       route("extra", 8, 200, 1), route("far", 40, 50, 0)]
        self.state = {
            "request_id": "menu-expansion-test", "user_request": "Find chicken rice",
            "parsed_request": ParsedRequest(
                intent="food_discovery", menu_item_preferences=["chicken rice"],
            ).model_dump(),
            "location_result": {
                "selected_centres": [selection("direct", 1), selection("fast", 2)],
                "logistics": self.routes, "confidence": 0.97,
                "reasoning": "Location preferred direct before fast for a supported travel trade-off.",
                "limitations": ["Original Location limitation."],
            },
        }
        self.matches = {name: [{"menu_item_id": name + ":rice"}]
                        for name in ("direct", "fast", "extra", "far")}
        self.menu = self.patches.enter_context(patch(
            "services.hawker_service.find_dietary_stalls",
            side_effect=lambda **kwargs: self.matches.get(kwargs["candidate_centre_ids"][0], []),
        ))
        self.budget = self.patches.enter_context(patch(
            "services.hawker_service.find_stalls_within_budget",
            side_effect=lambda **kwargs: self.matches.get(kwargs["candidate_centre_ids"][0], []),
        ))

    def expand(self):
        before = copy.deepcopy(self.state)
        result = orchestrator.expand_menu_search(self.state)
        self.assertEqual(self.state, before, "Expansion must not mutate its input state")
        return result.get("location_result")

    @staticmethod
    def ids(location):
        return [item["centre_id"] for item in location["selected_centres"]]

    def test_both_matches_preserve_slower_first_order_metadata_and_reasoning(self):
        result = self.expand()
        self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"])
        self.assertEqual(self.ids(result), ["direct", "fast"])
        self.assertEqual(result["reasoning"], self.state["location_result"]["reasoning"])
        self.assertEqual(result["confidence"], 0.97)
        self.assertEqual(result["logistics"], self.routes)
        self.assertEqual(self.menu.call_count, 2)
        self.assertNotIn("initial_location_reasoning", result["search_expansion"])

    def test_one_original_match_is_sufficient_without_topping_up(self):
        for matching in ("direct", "fast"):
            with self.subTest(matching=matching):
                self.menu.reset_mock()
                self.matches["direct"] = []
                self.matches["fast"] = []
                self.matches[matching] = [{"menu_item_id": matching + ":rice"}]
                result = self.expand()
                self.assertEqual(self.ids(result), [matching])
                original = next(item for item in self.state["location_result"]["selected_centres"]
                                if item["centre_id"] == matching)
                self.assertEqual(result["selected_centres"][0], {**original, "rank": 1})
                self.assertEqual(self.menu.call_count, 2)
                self.assertEqual(result["search_expansion"]["initial_location_reasoning"],
                                 self.state["location_result"]["reasoning"])
                self.assertIn("Menu filtering", result["reasoning"])

    def test_no_initial_matches_use_existing_bounded_supplementary_search(self):
        self.matches["direct"] = self.matches["fast"] = []
        self.routes.extend(route("other" + str(i), 15 + i) for i in range(5))
        self.matches.update({"other" + str(i): [{"menu_item_id": str(i)}] for i in range(5)})
        result = self.expand()
        self.assertEqual(self.ids(result), ["extra", "other0", "other1"])
        self.assertEqual(result["search_expansion"]["outcome"], "expanded_match_found")
        self.assertEqual([item["rank"] for item in result["selected_centres"]], [1, 2, 3])
        for item in result["selected_centres"]:
            self.assertIn("selected deterministically", item["rationale"])
        self.assertIn("supplementary", result["reasoning"])
        self.assertNotEqual(result["reasoning"], self.state["location_result"]["reasoning"])
        self.assertTrue(any("confidence does not assess" in note for note in result["limitations"]))

    def test_no_food_filter_skips_menu_work(self):
        self.state["parsed_request"]["menu_item_preferences"] = []
        self.assertIsNone(self.expand())
        self.menu.assert_not_called()

    def test_raw_walking_and_transfer_words_do_not_override_location_order(self):
        for text in ("Fastest chicken rice; I don't mind walking", "Chicken rice with few transfers"):
            with self.subTest(text=text):
                self.state["user_request"] = text
                result = self.expand()
                self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"])

    def test_supplements_obey_hard_limit_and_stay_in_existing_geographic_pool(self):
        self.matches["direct"] = self.matches["fast"] = []
        self.routes.append(route("too_slow", 11))
        self.matches["too_slow"] = [{"menu_item_id": "slow:rice"}]
        self.state["parsed_request"].update(max_travel_time_min=10, requested_area="east")
        result = self.expand()
        self.assertEqual(self.ids(result), ["extra"])
        self.assertEqual(result["logistics"], self.routes)
        self.assertTrue(all(item["travel"]["total_duration_min"] <= 10
                            for item in self.routes if item["centre_id"] in self.ids(result)))
        checked = [call.kwargs["candidate_centre_ids"][0] for call in self.menu.call_args_list]
        self.assertEqual(set(checked), {"fast", "extra"})

    def test_matching_original_outside_hard_limit_is_not_retained(self):
        self.state["parsed_request"]["max_travel_time_min"] = 10
        result = self.expand()
        self.assertEqual(self.ids(result), ["fast"])
        self.assertEqual(result["selected_centres"][0]["logistics_fit"], selection("fast", 2)["logistics_fit"])
        self.assertNotIn("Location preferred", result["reasoning"])

    def test_duplicate_routes_and_choices_do_not_consume_selection_slots(self):
        self.routes.append(copy.deepcopy(self.routes[0]))
        self.state["location_result"]["selected_centres"].insert(1, selection("direct", 2))
        result = self.expand()
        self.assertEqual(self.ids(result), ["direct", "fast"])
        self.assertEqual([item["rank"] for item in result["selected_centres"]], [1, 2])
        self.assertEqual(result["search_expansion"]["initial_location_reasoning"],
                         self.state["location_result"]["reasoning"])

    def test_supplementary_duplicate_routes_are_not_selected_twice(self):
        self.matches["direct"] = self.matches["fast"] = []
        self.routes.append(copy.deepcopy(self.routes[2]))
        self.assertEqual(self.ids(self.expand()), ["extra"])

    def test_conflicting_primary_evidence_is_excluded_in_either_order(self):
        variants = (
            ("duration", "travel", "total_duration_min", 25),
            ("transfers", "travel", "transfers", 4),
            ("walking", "travel", "walking_distance_m", 100),
            ("mode", "travel", "mode", "walk"),
            ("coordinates", None, "coordinates", {"lat": 1.34, "lng": 103.9}),
            ("address", None, "address", "Different destination"),
            ("name", None, "centre_name", "Different centre"),
            ("route identity", None, "route_id", "different:primary"),
        )
        for name, parent, key, value in variants:
            previous = None
            for reverse in (False, True):
                with self.subTest(field=name, reverse=reverse):
                    original = copy.deepcopy(self.routes[0])
                    conflicting = copy.deepcopy(original)
                    (conflicting[parent] if parent else conflicting)[key] = value
                    pair = [conflicting, original] if reverse else [original, conflicting]
                    self.state["location_result"]["logistics"] = pair + self.routes[1:]
                    self.menu.reset_mock()
                    result = self.expand()
                    self.assertEqual(self.ids(result), ["fast"])
                    self.assertEqual(result["selected_centres"][0], {**selection("fast", 2), "rank": 1})
                    self.assertNotEqual(result["reasoning"], self.state["location_result"]["reasoning"])
                    self.assertEqual(result["search_expansion"]["initial_location_reasoning"],
                                     self.state["location_result"]["reasoning"])
                    self.assertTrue(any("Ambiguous or unusable" in note for note in result["limitations"]))
                    checked = [call.kwargs["candidate_centre_ids"][0] for call in self.menu.call_args_list]
                    self.assertNotIn("direct", checked)
                    decision = (result["selected_centres"], result["reasoning"], result["search_expansion"])
                    if previous is not None:
                        self.assertEqual(decision, previous)
                    previous = decision

    def test_conflicting_duplicates_cannot_cross_hard_limit(self):
        self.state["parsed_request"]["max_travel_time_min"] = 10
        for pair in ((12, 6), (6, 12)):
            with self.subTest(durations=pair):
                self.state["location_result"]["logistics"] = [
                    route("direct", minutes) for minutes in pair
                ] + self.routes[1:]
                result = self.expand()
                self.assertEqual(self.ids(result), ["fast"])
                self.assertEqual(result["search_expansion"]["search_horizon_min"], 10)
                self.assertIn("initial_location_reasoning", result["search_expansion"])

    def test_unusable_duplicate_cannot_disappear_before_conflict_detection(self):
        for bad_travel in (None, {"source": "unavailable", "total_duration_min": 12},
                           {"source": "onemap", "mode": "pt", "total_duration_min": None}):
            for reverse in (False, True):
                with self.subTest(travel=bad_travel, reverse=reverse):
                    good = copy.deepcopy(self.routes[0])
                    bad = {**copy.deepcopy(good), "travel": bad_travel}
                    pair = [bad, good] if reverse else [good, bad]
                    self.state["location_result"]["logistics"] = pair + self.routes[1:]
                    self.assertEqual(self.ids(self.expand()), ["fast"])

    def test_unusable_source_mode_and_duration_cannot_retain_location_metadata(self):
        cases = [("source", source) for source in ("unavailable", "other_provider", None)]
        cases += [("mode", "teleport")]
        cases += [("total_duration_min", duration) for duration in (
            None, -1, float("nan"), float("inf"), float("-inf"), "12", True,
        )]
        for key, value in cases:
            with self.subTest(field=key, value=value):
                bad = copy.deepcopy(self.routes[0])
                bad["travel"][key] = value
                self.state["location_result"]["logistics"] = [bad] + self.routes[1:]
                result = self.expand()
                self.assertEqual(self.ids(result), ["fast"])
                self.assertEqual(result["selected_centres"][0], {**selection("fast", 2), "rank": 1})
                self.assertEqual(result["search_expansion"]["initial_location_reasoning"],
                                 self.state["location_result"]["reasoning"])

    def test_unsafe_routes_are_not_used_as_supplements_or_clarification_context(self):
        self.matches["direct"] = self.matches["fast"] = []
        unsafe = copy.deepcopy(self.routes[2])
        unsafe["travel"]["source"] = "unavailable"
        self.state["location_result"]["logistics"] = self.routes[:2] + [unsafe]
        result = self.expand()
        self.assertNotIn("extra", self.ids(result))
        self.assertEqual(result["search_expansion"]["outcome"], "no_match_in_routed_pool")
        self.assertTrue(all(call.kwargs["candidate_centre_ids"] != ["extra"]
                            for call in self.menu.call_args_list))
        self.state["location_result"]["logistics"] = self.routes[:3] + [route("extra", 9)]
        self.assertNotIn("extra", self.ids(self.expand()))

    def test_all_supplied_routes_unsafe_clear_selection_and_preserve_provenance(self):
        unavailable = copy.deepcopy(self.routes[0])
        unavailable["travel"]["source"] = "unavailable"
        missing = copy.deepcopy(self.routes[0])
        missing["travel"].pop("total_duration_min")
        pools = ([self.routes[0], route("direct", 25)], [unavailable], [missing],
                 [None, {}, {"centre_id": "direct", "travel": None}])
        for pool in pools:
            with self.subTest(pool=pool):
                self.state["location_result"]["logistics"] = pool
                self.menu.reset_mock()
                token, events = begin_activity_capture()
                try:
                    result = self.expand()
                finally:
                    end_activity_capture(token)
                self.assertEqual(result["selected_centres"], [])
                self.assertEqual(result["confidence"], 0.0)
                self.assertIn("ambiguous or unusable", result["reasoning"])
                self.assertEqual(result["search_expansion"]["initial_location_reasoning"],
                                 self.state["location_result"]["reasoning"])
                self.assertEqual(result["search_expansion"]["route_candidate_pool_size"], 0)
                self.assertEqual(events[-1]["event"], "menu_search.completed")
                self.assertEqual(events[-1]["request_id"], self.state["request_id"])
                self.assertEqual(events[-1]["selected_centres"], [])
                self.menu.assert_not_called()

    def test_all_unsafe_pool_cannot_hide_behind_absent_food_filter(self):
        self.state["parsed_request"]["menu_item_preferences"] = []
        self.state["location_result"]["logistics"] = [route("direct", float("nan"))]
        result = self.expand()
        self.assertEqual(result["selected_centres"], [])
        self.assertIn("no centre could safely be selected", result["reasoning"])
        self.menu.assert_not_called()

    def test_unverifiable_dietary_branch_cannot_retain_unsafe_choices(self):
        self.state["parsed_request"]["dietary_preferences"] = ["vegetarian"]
        self.state["location_result"]["logistics"] = [self.routes[0], route("direct", 25)]
        result = self.expand()
        self.assertEqual(result["selected_centres"], [])
        self.assertEqual(result["search_expansion"]["outcome"], "dietary_verification_unavailable")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["search_expansion"]["initial_location_reasoning"],
                         self.state["location_result"]["reasoning"])
        self.assertIn("Route evidence", result["reasoning"])
        self.menu.assert_not_called()

    def test_later_identical_copy_cannot_restore_an_ambiguous_centre(self):
        good = self.routes[0]
        bad = route("direct", 25)
        for copies in ([good, bad, good], [bad, good, bad], [good, good, bad]):
            with self.subTest(durations=[item["travel"]["total_duration_min"] for item in copies]):
                self.state["location_result"]["logistics"] = copies + self.routes[1:]
                self.assertEqual(self.ids(self.expand()), ["fast"])

    def test_genuinely_empty_or_missing_pool_keeps_skip_behavior(self):
        for value in ([], None):
            with self.subTest(pool=value):
                self.state["location_result"]["logistics"] = value
                self.assertIsNone(self.expand())
        self.state["location_result"].pop("logistics")
        self.assertIsNone(self.expand())
        self.menu.assert_not_called()

    def test_zero_duration_is_usable_with_zero_hard_limit(self):
        self.state["location_result"]["logistics"] = [route("direct", 0)] + self.routes[1:]
        self.state["parsed_request"]["max_travel_time_min"] = 0
        result = self.expand()
        self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"][:1])

    def test_alternative_route_differences_do_not_conflict_with_primary_evidence(self):
        duplicate = copy.deepcopy(self.routes[0])
        duplicate["alternative_travel"] = {"source": "onemap", "total_duration_min": 30, "mode": "walk"}
        self.state["location_result"]["logistics"] = self.routes + [duplicate]
        result = self.expand()
        self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"])
        self.assertEqual(result["reasoning"], self.state["location_result"]["reasoning"])

    def test_no_matches_preserves_existing_clarification_outcome(self):
        self.matches = {}
        result = self.expand()
        self.assertEqual(result["search_expansion"]["outcome"], "no_match_in_routed_pool")
        self.assertLessEqual(len(self.ids(result)), 3)
        self.assertIn("No source-backed menu match", result["reasoning"])

    def test_unchanged_clarification_selection_keeps_original_reasoning(self):
        self.matches = {}
        self.state["location_result"]["selected_centres"] = [
            selection("fast", 1), selection("extra", 2), selection("direct", 3),
        ]
        self.state["location_result"]["reasoning"] = "Initial duration order: fast, extra, direct."
        result = self.expand()
        self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"])
        self.assertEqual(result["reasoning"], self.state["location_result"]["reasoning"])
        self.assertEqual(result["search_expansion"]["outcome"], "no_match_in_routed_pool")

    def test_original_matching_choices_are_capped_at_three(self):
        self.state["user_request"] = "Chicken rice; willing to travel farther"
        self.state["location_result"]["selected_centres"] = [
            selection(name, rank) for rank, name in enumerate(("direct", "fast", "extra", "far"), 1)
        ]
        result = self.expand()
        self.assertEqual(self.ids(result), ["direct", "fast", "extra"])
        self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"][:3])

    def test_farther_match_requires_approval_without_bypassing_horizon(self):
        self.matches = {"far": [{"menu_item_id": "far:rice"}]}
        result = self.expand()
        self.assertEqual(result["search_expansion"]["outcome"], "farther_match_requires_approval")
        self.assertNotIn("far", self.ids(result))
        self.assertEqual(result["search_expansion"]["farther_matches"][0]["centre_id"], "far")

    def test_explicit_farther_permission_preserves_existing_search_behavior(self):
        self.matches = {"far": [{"menu_item_id": "far:rice"}]}
        self.state["user_request"] = "Chicken rice; willing to travel farther"
        self.assertEqual(self.ids(self.expand()), ["far"])

    def test_farther_permission_cannot_override_hard_limit(self):
        self.matches = {"far": [{"menu_item_id": "far:rice"}]}
        self.state["user_request"] = "Chicken rice; willing to travel farther"
        self.state["parsed_request"]["max_travel_time_min"] = 20
        result = self.expand()
        self.assertEqual(result["search_expansion"]["outcome"], "no_match_in_routed_pool")
        self.assertNotIn("far", self.ids(result))

    def test_empty_selection_or_missing_routes_preserves_skip_behavior(self):
        for key in ("selected_centres", "logistics"):
            with self.subTest(key=key):
                before = copy.deepcopy(self.state["location_result"])
                self.state["location_result"][key] = []
                self.assertIsNone(self.expand())
                self.state["location_result"] = before
        self.menu.assert_not_called()

    def test_unusable_ranking_keeps_legacy_unranked_walking_sort(self):
        self.state["location_result"]["selected_centres"] = [selection("not_in_route_pool", 1)]
        self.state["user_request"] = "Chicken rice with least walking"
        result = self.expand()
        self.assertEqual(self.ids(result), ["fast", "extra", "direct"])

    def test_existing_guard_branches_still_skip(self):
        for key in ("needs_location", "location_reused", "max_queue_min", "intent"):
            with self.subTest(key=key):
                before = copy.deepcopy(self.state)
                if key == "max_queue_min":
                    self.state["parsed_request"][key] = 10
                elif key == "intent":
                    self.state["parsed_request"][key] = "directions"
                else:
                    self.state[key] = True
                self.assertIsNone(self.expand())
                self.state = before
        self.menu.assert_not_called()

    def test_unverifiable_dietary_filter_preserves_metadata_and_updates_changed_reasoning(self):
        self.state["parsed_request"].update(dietary_preferences=["vegetarian"], max_travel_time_min=10)
        result = self.expand()
        self.assertEqual(self.ids(result), ["fast"])
        self.assertEqual(result["selected_centres"][0], {**selection("fast", 2), "rank": 1})
        self.assertEqual(result["search_expansion"]["outcome"], "dietary_verification_unavailable")
        self.assertIn("cannot verify", result["reasoning"])
        self.menu.assert_not_called()

    def test_unverifiable_dietary_filter_keeps_unchanged_reasoning(self):
        self.state["parsed_request"]["dietary_preferences"] = ["vegetarian"]
        result = self.expand()
        self.assertEqual(result["selected_centres"], self.state["location_result"]["selected_centres"])
        self.assertEqual(result["reasoning"], self.state["location_result"]["reasoning"])
        self.menu.assert_not_called()

    def test_real_food_matcher_and_budget_keep_existing_eligibility_policy(self):
        records = [{
            "menu_item_id": name + ":rice", "hawker_centre_id": name,
            "menu_item_name": "Chicken rice", "stall_name": "Rice Stall",
            "categories": ["Chinese"], "dietary_tags": ["healthy"],
            "price_sgd_min": price, "price_sgd_max": price,
        } for name, price in (("direct", 4), ("fast", 7), ("extra", 3))]
        self.state["parsed_request"].update(
            budget_amount=5, cuisine_preferences=["Chinese"],
            dietary_preferences=["healthy"], stall_name_preferences=["Rice Stall"],
            menu_item_preferences=["chicken", "rice"],
        )
        self.menu.side_effect = find_dietary_stalls
        self.budget.side_effect = None
        self.budget.return_value = [item for item in records if item["price_sgd_min"] <= 5]
        with patch("services.hawker_service._filtered_menu", side_effect=lambda **kwargs: [
            item for item in records if item["hawker_centre_id"] in kwargs["candidate_centre_ids"]
        ]):
            result = self.expand()
        self.assertEqual(self.ids(result), ["direct"])
        self.assertEqual(self.menu.call_count, 2)

    def test_budget_intersection_and_all_food_filters_are_forwarded(self):
        self.state["parsed_request"].update(
            budget_amount=5, cuisine_preferences=["Chinese"],
            stall_name_preferences=["Rice Stall"], excluded_menu_item_ids=["excluded"],
        )
        self.budget.side_effect = lambda **kwargs: self.matches["fast"]
        result = self.expand()
        self.assertEqual(self.ids(result), ["fast"])
        for call in self.menu.call_args_list:
            self.assertEqual(call.kwargs["menu_item_preferences"], ["chicken rice"])
            self.assertEqual(call.kwargs["cuisine_preferences"], ["Chinese"])
            self.assertEqual(call.kwargs["stall_name_preferences"], ["Rice Stall"])
            self.assertEqual(call.kwargs["excluded_menu_item_ids"], ["excluded"])
        self.assertTrue(all(call.kwargs["budget"] == 5 for call in self.budget.call_args_list))

    def test_trace_correlation_and_public_location_contract(self):
        token, events = begin_activity_capture()
        try:
            result = self.expand()
        finally:
            end_activity_capture(token)
        self.assertEqual(events[-1]["event"], "menu_search.completed")
        self.assertEqual(events[-1]["request_id"], self.state["request_id"])
        self.assertEqual([item["centre_id"] for item in events[-1]["selected_centres"]], ["direct", "fast"])
        self.assertEqual(events[-1]["added_centre_ids"], [])
        self.assertFalse(events[-1]["search_expanded"])
        decision = LocationDecision.model_validate(result)
        self.assertEqual([item.centre_id for item in decision.selected_centres], ["direct", "fast"])

    def test_worker_dispatch_and_verifier_handoff_keep_expanded_selection(self):
        result = self.expand()
        state = {**self.state, "location_result": result, "agent_results": [
            {"agent": "location", "payload": self.state["location_result"]},
        ], "tasks": [PlannedTask(agent="dietary", task="Find food")]}
        dispatch = orchestrator.dispatch_workers(state)
        self.assertEqual(dispatch[0].arg["location_result"]["selected_centres"], result["selected_centres"])
        with patch("agents.verifier._verify_results_with_context", return_value=([], None)) as verify:
            orchestrator.verify_results_node(state)
        envelope = verify.call_args.args[0][0]
        self.assertEqual(envelope["payload"]["selected_centres"], result["selected_centres"])
        self.assertEqual(envelope["payload"]["reasoning"], result["reasoning"])


if __name__ == "__main__":
    unittest.main()
