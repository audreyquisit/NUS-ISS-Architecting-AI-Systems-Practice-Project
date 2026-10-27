"""Structured dish/cuisine authority with real local matching and verification."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from agents import dietary_agent, verifier
from evaluations.pilot_support import build_routes
from models.location_models import LocationDecision, RankedCentre
from services import hawker_service
from services.location_service import load_hawker_centres


class DietaryPreferenceAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": ""}))
        self.llm = self.patches.enter_context(patch.object(dietary_agent, "_llm_assess",
            side_effect=AssertionError("No model calls")))
        self.menu = [dict(hawker_service.get_menu_item_record(key)) for key in (
            "wakwak_37_item_005", "wakwak_448_item_000", "wakwak_881_item_000")]
        self.patches.enter_context(patch.object(hawker_service, "_menu_catalogue", return_value=tuple(self.menu)))
        self.scope = {item["hawker_centre_id"] for item in self.menu}

    def run_agent(self, **fields):
        result = dietary_agent.run("Find catalogue food", user_request="Find chicken rice",
            candidate_centre_ids=list(self.scope), **fields)
        self.llm.assert_not_called()
        return result

    def test_explicit_dish_does_not_add_inferred_cuisine(self):
        result = self.run_agent(menu_item_preferences=["chicken rice"])
        self.assertEqual(result["preferences"]["cuisine"], [])
        self.assertEqual({row["menu_item_id"] for row in result["candidates"]},
            {row["menu_item_id"] for row in self.menu})

    def test_explicit_malay_cuisine_is_not_augmented_with_inferred_rice(self):
        result = self.run_agent(cuisine_preferences=["malay"])
        self.assertEqual(result["preferences"]["cuisine"], ["malay"])
        self.assertEqual([row["menu_item_id"] for row in result["candidates"]], ["wakwak_881_item_000"])

    def test_explicit_chinese_cuisine_still_rejects_category_disagreement(self):
        result = self.run_agent(menu_item_preferences=["chicken rice"], cuisine_preferences=["chinese"])
        self.assertEqual(result["preferences"]["cuisine"], ["chinese"])
        self.assertEqual({row["menu_item_id"] for row in result["candidates"]},
            {"wakwak_37_item_005", "wakwak_448_item_000"})

    def test_legacy_unstructured_request_keeps_inferred_cuisine(self):
        result = self.run_agent()
        self.assertEqual(result["preferences"]["cuisine"], ["rice"])
        self.assertNotIn("wakwak_881_item_000", {row["menu_item_id"] for row in result["candidates"]})

    def test_maxwell_dish_survives_actual_verification_and_join(self):
        order = ["maxwell_food_centre", "newton_food_centre",
                 "buffalo_road_blk_665_tekka_centre_zhu_jiao_market"]
        location = LocationDecision(selected_centres=[RankedCentre(
            centre_id=centre, rank=index, logistics_fit=0.5, rationale="Untrusted worker prose")
            for index, centre in enumerate(order, 1)], logistics=build_routes(load_hawker_centres()),
            reasoning="Untrusted worker prose", confidence=0.5).model_dump()
        results = verifier.verify_results([{"agent": "location", "payload": location},
            self.run_agent(menu_item_preferences=["chicken rice"])],
            {"intent": "food_discovery", "menu_item_preferences": ["chicken rice"], "travel_mode": "pt"})
        joined = next(row["candidates"] for row in results if row["agent"] == "candidate_join")
        self.assertEqual([row["hawker_centre_id"] for row in joined], order)
        self.assertEqual(joined[0]["menu_item_id"], "wakwak_881_item_000")

    def test_dietary_restrictions_exclusions_and_budget_still_fail_closed(self):
        for constraint in ({"dietary_preferences": ["vegetarian"]},
                           {"dietary_preferences": ["no-beef"]},
                           {"excluded_menu_item_ids": [item["menu_item_id"] for item in self.menu]},
                           {"budget_amount": 2}):
            with self.subTest(constraint=constraint):
                agent_fields = {key: value for key, value in constraint.items() if key != "budget_amount"}
                result = self.run_agent(menu_item_preferences=["chicken rice"], **agent_fields)
                request = {"intent": "food_discovery", "menu_item_preferences": ["chicken rice"], **constraint}
                allowed = verifier._matching_menu_ids(request, self.scope)
                checked = verifier._verify_worker_result(result, request,
                    approved_centres=self.scope, allowed_menu_ids=allowed)
                self.assertEqual(checked["candidates"], [])


if __name__ == "__main__":
    unittest.main()
