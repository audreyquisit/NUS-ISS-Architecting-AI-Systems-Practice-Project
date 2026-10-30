"""Dietary output boundary: real local SDK replies, no external services."""

import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import httpx
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, ContentFilterFinishReasonError
from pydantic import ValidationError

from agents import dietary_agent as agent, verifier
from services import hawker_service


class DietaryStructuredOutputTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test"}))
        self.source = dict(hawker_service.get_menu_item_record("wakwak_588_item_004"))
        self.sources = [self.source]
        self.patches.enter_context(patch.object(agent, "find_dietary_stalls", side_effect=lambda **_: copy.deepcopy(self.sources)))
        self.requests = []

    def assessment(self, **fields):
        return {"menu_item_id": self.source["menu_item_id"], "dietary_suitable": False,
                "confidence": 0.2, "reason": "Ingredients and certification are unavailable.", **fields}

    def run_sdk(self, assessments=None, *, message=None, arguments=None, status=200, finish="stop", transport_error=None):
        body = {"assessments": [self.assessment()] if assessments is None else assessments}
        before = copy.deepcopy(self.sources)

        def response(request):
            self.requests.append(request)
            sent = json.loads(request.content)
            self.assertEqual(sent["model"], "gpt-4o-mini")
            self.assertEqual(sent["tools"][0]["function"]["name"], "_DietaryAssessments")
            if transport_error:
                raise transport_error
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "private-provider-detail", "type": "error"}})
            tool_message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "offline-call", "type": "function", "function": {
                    "name": "_DietaryAssessments", "arguments": json.dumps(body) if arguments is None else arguments}}]}
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 1,
                "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": finish,
                    "message": tool_message if message is None else message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        def factory(**kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            return ChatOpenAI(http_client=client, **kwargs)

        with httpx.Client(transport=httpx.MockTransport(response), trust_env=False) as client, \
                patch("langchain_openai.ChatOpenAI", side_effect=factory) as constructor:
            try:
                result = agent.run("Find vegetarian food", user_request="Find vegetarian food",
                                   dietary_preferences=["vegetarian"])
            finally:
                constructor.assert_called_once()
                self.assertEqual(self.sources, before)
        return result

    def assert_safe_fallback(self, result):
        self.assertTrue(any("fallback logic was used" in text for text in result["limitations"]))
        self.assertTrue(all(row["dietary_suitable"] is False for row in result["candidates"]))
        self.assertNotIn("private-provider-detail", str(result))
        self.assertNotIn("tool_calls", str(result))

    def test_valid_strict_boolean_values_and_source_preservation(self):
        for value in (False, True):
            with self.subTest(value=value):
                result = self.run_sdk([self.assessment(dietary_suitable=value)])
                row = result["candidates"][0]
                self.assertIs(row["dietary_suitable"], value)
                for key, source_value in self.source.items():
                    if key not in {"reason", "dietary_suitable", "dietary_confidence"}:
                        self.assertEqual(row[key], source_value)
                self.assertFalse(any("fallback logic" in x for x in result["limitations"]))
        self.assertEqual(len(self.requests), 2)

    def test_string_and_numeric_boolean_values_rejected(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value):
                self.assert_safe_fallback(self.run_sdk([self.assessment(dietary_suitable=value)]))

    def test_confidence_is_finite_bounded_and_not_coerced(self):
        for value in (float("nan"), float("inf"), -float("inf"), -0.1, 1.1, "0.8", True, None):
            with self.subTest(value=value):
                self.assert_safe_fallback(self.run_sdk([self.assessment(confidence=value)]))
        for value in (0, 1):
            self.assertEqual(self.run_sdk([self.assessment(confidence=value)])["candidates"][0]["dietary_confidence"], value)

    def test_unknown_missing_and_malformed_ids_rejected(self):
        missing = self.assessment()
        del missing["menu_item_id"]
        for assessment in (missing, self.assessment(menu_item_id="unknown"),
                           self.assessment(menu_item_id=[]), self.assessment(menu_item_id="")):
            with self.subTest(assessment=assessment):
                self.assert_safe_fallback(self.run_sdk([assessment]))

    def test_missing_assessments_stay_unsuitable(self):
        result = self.run_sdk([])
        self.assertIs(result["candidates"][0]["dietary_suitable"], False)
        self.assertEqual(result["candidates"][0]["dietary_confidence"], 0)

    def test_partial_assessment_does_not_promote_unassessed_candidate(self):
        self.sources.append({**self.source, "menu_item_id": "unassessed"})
        result = self.run_sdk([self.assessment(dietary_suitable=True)])
        self.assertIs(result["candidates"][0]["dietary_suitable"], True)
        self.assertIs(result["candidates"][1]["dietary_suitable"], False)
        self.assertEqual(result["candidates"][1]["dietary_confidence"], 0)

    def test_identical_duplicates_collapse(self):
        row = self.assessment()
        result = self.run_sdk([row, copy.deepcopy(row)])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertFalse(any("fallback logic" in x for x in result["limitations"]))

    def test_conflicting_duplicates_fail_closed_in_either_order(self):
        first = self.assessment(dietary_suitable=True)
        for second in (self.assessment(), self.assessment(dietary_suitable=True, confidence=0.9),
                       self.assessment(dietary_suitable=True, reason="Different claim")):
            for pair in ([first, second], [second, first]):
                self.assert_safe_fallback(self.run_sdk(pair))

    def test_invalid_containers_and_list_bound(self):
        for body in ([], None, {"assessments": {}}, {"assessments": None}, {"assessments": ["invalid"]},
                     {"assessments": [self.assessment()] * 101}):
            with self.subTest(body=body):
                self.assert_safe_fallback(self.run_sdk(arguments=json.dumps(body)))

    def test_model_owned_identity_or_certification_fields_rejected(self):
        for field in ("stall_id", "hawker_centre_id", "menu_item_name", "ingredients", "certification"):
            with self.subTest(field=field):
                self.assert_safe_fallback(self.run_sdk([self.assessment(**{field: "invented"})]))

    def test_fenced_empty_and_non_json_tool_arguments(self):
        for content in ("", "not JSON", "```json\n{}\n```"):
            self.assert_safe_fallback(self.run_sdk(arguments=content))

    def test_plain_prose_empty_content_and_content_blocks_do_not_bypass_schema(self):
        for content in ("Unverified assertion", "", [{"type": "text", "text": "Vegetarian certified"}]):
            with self.subTest(content=content):
                self.assert_safe_fallback(self.run_sdk(message={"role": "assistant", "content": content}))

    def test_sdk_refusal_content_filter_and_length_rejection(self):
        self.assert_safe_fallback(self.run_sdk(message={"role": "assistant", "content": None, "refusal": "Cannot comply"}))
        for finish in ("content_filter", "length"):
            self.assert_safe_fallback(self.run_sdk(message={"role": "assistant", "content": None}, finish=finish))

    def test_transport_and_server_errors_fallback_without_retry(self):
        for status in (429, 500, 503):
            before = len(self.requests)
            self.assert_safe_fallback(self.run_sdk(status=status))
            self.assertEqual(len(self.requests), before + 1)
        for error in (httpx.ConnectError("private-provider-detail"), httpx.ReadTimeout("private-provider-detail")):
            before = len(self.requests)
            self.assert_safe_fallback(self.run_sdk(transport_error=error))
            self.assertEqual(len(self.requests), before + 1)

    def test_authentication_errors_propagate(self):
        with self.assertRaises(AuthenticationError):
            self.run_sdk(status=401)

    def test_typed_content_filter_and_parser_errors_are_recoverable(self):
        for error in (ContentFilterFinishReasonError(), OutputParserException("private-provider-detail")):
            model = Mock()
            model.with_structured_output.return_value = RunnableLambda(lambda _: self.raise_error(error))
            with patch("langchain_openai.ChatOpenAI", return_value=model):
                self.assert_safe_fallback(agent.run("Find vegetarian food", dietary_preferences=["vegetarian"]))

    def test_programming_and_setup_errors_propagate(self):
        for error in (RuntimeError("programming"), TypeError("programming"), AttributeError("programming")):
            model = Mock()
            model.with_structured_output.return_value = RunnableLambda(lambda _: self.raise_error(error))
            with patch("langchain_openai.ChatOpenAI", return_value=model), self.assertRaises(type(error)):
                agent.run("Find vegetarian food", dietary_preferences=["vegetarian"])
        for boundary in ("constructor", "schema_setup"):
            model = Mock()
            model.with_structured_output.side_effect = ValueError("configuration")
            with patch("langchain_openai.ChatOpenAI", side_effect=ValueError("configuration") if boundary == "constructor" else None,
                       return_value=model), self.assertRaises(ValueError):
                agent.run("Find vegetarian food", dietary_preferences=["vegetarian"])

    @staticmethod
    def raise_error(error):
        raise error

    def test_schema_validation_error_at_invocation_is_recoverable(self):
        try:
            agent._DietaryAssessments.model_validate({"assessments": None})
        except ValidationError as error:
            model = Mock()
            model.with_structured_output.return_value = RunnableLambda(lambda _: self.raise_error(error))
            with patch("langchain_openai.ChatOpenAI", return_value=model):
                self.assert_safe_fallback(agent.run("Find vegetarian food", dietary_preferences=["vegetarian"]))

    def test_only_sent_candidates_can_be_assessed_and_unsent_are_negative(self):
        self.sources = [{**self.source, "menu_item_id": "fixture_" + str(index)} for index in range(101)]
        result = self.run_sdk([self.assessment(menu_item_id="fixture_0")])
        self.assertTrue(all(row["dietary_suitable"] is False for row in result["candidates"]))
        self.assert_safe_fallback(self.run_sdk([self.assessment(menu_item_id="fixture_100", dietary_suitable=True)]))

    def test_positive_model_assertion_does_not_certify_vegetarian_food(self):
        result = self.run_sdk([self.assessment(dietary_suitable=True)])
        request = {"intent": "food_discovery", "dietary_preferences": ["vegetarian"]}
        scope = {self.source["hawker_centre_id"]}
        allowed = verifier._matching_menu_ids(request, scope)
        checked = verifier._verify_worker_result(result, request, approved_centres=scope, allowed_menu_ids=allowed)
        self.assertEqual(checked["candidates"], [])
        self.assertTrue(any("cannot be verified" in x for x in result["limitations"]))


if __name__ == "__main__":
    unittest.main()
