"""No-network SDK/HTTP semantics and feasible evaluation fixtures."""

import gzip
import json
import unittest

import httpx
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, AuthenticationError, ContentFilterFinishReasonError
from pydantic import ValidationError

from agents.recommendation_rendering import _RecommendationPlan
from evaluations.pilot_support import ROUTE_FIXTURES, build_routes, response_observer


class PilotEvaluationSupportTests(unittest.TestCase):
    @staticmethod
    def payload(content=None, finish="stop"):
        return {"id": "offline-completion", "object": "chat.completion", "created": 1,
            "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": finish,
                "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}

    def test_original_decoded_response_reconstruction_reproduces_failure(self):
        compressed = gzip.compress(json.dumps({"message": "Laksa — S$5"}).encode())
        with httpx.Client(transport=httpx.MockTransport(lambda request:
                httpx.Response(200, headers={"content-encoding": "gzip"}, content=compressed)),
                trust_env=False) as client:
            original = client.get("https://offline.invalid")
        self.assertEqual(original.json(), {"message": "Laksa — S$5"})
        with self.assertRaises(httpx.DecodingError):
            httpx.Response(200, headers=original.headers, content=original.read())

    def invoke(self, payload, status=200):
        records, requests = [], []

        def response(request):
            requests.append(request)
            return httpx.Response(status, headers={"content-encoding": "gzip"},
                content=gzip.compress(json.dumps(payload).encode()))

        with httpx.Client(transport=httpx.MockTransport(response), trust_env=False,
                event_hooks={"response": [response_observer(records)]}) as client:
            model = ChatOpenAI(model="gpt-4o", api_key="offline-test", max_retries=0, http_client=client)
            try:
                return model.with_structured_output(_RecommendationPlan, include_raw=True).invoke("Offline fixture")
            finally:
                self.assertEqual(len(requests), 1)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["status_code"], status)
                if status == 200:
                    self.assertEqual(records[0]["usage"], {
                        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})

    def test_original_wrapper_decoding_error_becomes_sdk_connection_error(self):
        requests = []

        def broken_wrapper(request):
            requests.append(request)
            original = httpx.Response(200, headers={"content-encoding": "gzip"},
                content=gzip.compress(json.dumps(self.payload()).encode()))
            return httpx.Response(200, headers=original.headers, content=original.read())

        with httpx.Client(transport=httpx.MockTransport(broken_wrapper), trust_env=False) as client:
            model = ChatOpenAI(model="gpt-4o", api_key="offline-test", max_retries=0, http_client=client)
            with self.assertRaises(APIConnectionError) as caught:
                model.with_structured_output(_RecommendationPlan, include_raw=True).invoke("Offline fixture")
        self.assertIsInstance(caught.exception.__cause__, httpx.DecodingError)
        self.assertEqual(len(requests), 1)

    def test_observer_preserves_compressed_success_through_actual_sdk(self):
        content = json.dumps({"choices": [{"menu_item_id": "menu:rice", "reason_refs": ["eligible_menu"]}]})
        result = self.invoke(self.payload(content))
        self.assertIsNone(result["parsing_error"])
        self.assertEqual(result["parsed"].choices[0].menu_item_id, "menu:rice")

    def test_observer_preserves_sdk_content_filter_exception(self):
        with self.assertRaises(ContentFilterFinishReasonError):
            self.invoke(self.payload(finish="content_filter"))

    def test_observer_preserves_schema_rejection(self):
        with self.assertRaises(ValidationError):
            self.invoke(self.payload('{"choices": [{"menu_item_id": []}]}'))

    def test_observer_preserves_authentication_failure(self):
        with self.assertRaises(AuthenticationError):
            self.invoke({"error": {"message": "Offline invalid credential fixture", "type": "invalid_request_error"}}, 401)

    def test_observer_leaves_non_json_content_and_headers_untouched(self):
        records = []
        with httpx.Client(transport=httpx.MockTransport(lambda request:
                httpx.Response(200, text="not JSON", headers={"x-fixture": "unchanged"})),
                event_hooks={"response": [response_observer(records)]}, trust_env=False) as client:
            result = client.get("https://offline.invalid")
        self.assertEqual(result.text, "not JSON")
        self.assertEqual(result.headers["x-fixture"], "unchanged")
        self.assertIsNone(records[0]["usage"])

    def test_corrected_routes_are_feasible_and_preserve_four_minute_tradeoff(self):
        catalogue = [{"id": centre, "name": centre, "address": "Synthetic fixture",
            "latitude": 1.33, "longitude": 103.85} for centre, *_ in ROUTE_FIXTURES]
        routes = {route.centre_id: route for route in build_routes(catalogue)}
        for route in routes.values():
            self.assertEqual(route.travel.mode, "pt")
            self.assertLessEqual(route.travel.walking_time_min, route.travel.total_duration_min)
            self.assertLessEqual(route.travel.total_duration_min, 30)
            self.assertGreater(route.travel.walking_distance_m, 0)
            self.assertGreaterEqual(route.travel.transfers, 0)
        fast, balanced = routes["newton_food_centre"].travel, routes["maxwell_food_centre"].travel
        self.assertEqual(balanced.total_duration_min - fast.total_duration_min, 4)
        self.assertLess(balanced.walking_distance_m, fast.walking_distance_m)
        self.assertLess(balanced.transfers, fast.transfers)


if __name__ == "__main__":
    unittest.main()
