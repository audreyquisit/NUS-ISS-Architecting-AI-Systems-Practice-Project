"""The weather service may only report a condition it actually read."""

import unittest
from unittest.mock import patch

from services import weather_service


FORECAST_RECORD = {
    "general": {
        "forecast": {"text": "Cloudy", "code": "100"},
        "temperature": {"high": 32, "low": 25},
        "validPeriod": {"text": "6 pm to 6 pm"},
    },
    "updatedTimestamp": "2026-10-20T08:00:00+08:00",
    "periods": [
        {
            "timePeriod": {
                "text": "6 pm - 6 am",
                "start": "2026-10-20T18:00:00+08:00",
                "end": "2026-10-21T06:00:00+08:00",
            },
            "regions": {
                "east": {"text": "Light rain", "code": "120"},
                "west": {"text": "Sunny", "code": "010"},
            },
        }
    ],
}


class WeatherServiceTests(unittest.TestCase):
    def setUp(self):
        weather_service._CACHE["payload"] = None
        weather_service._CACHE["at"] = 0.0

    def tearDown(self):
        weather_service._CACHE["payload"] = None
        weather_service._CACHE["at"] = 0.0

    def test_an_unreadable_forecast_reports_no_condition(self):
        with patch.object(
            weather_service.requests, "get", side_effect=weather_service.requests.RequestException("offline")
        ):
            result = weather_service.get_weather(area="east")
        self.assertEqual(result["condition"], "unavailable")
        self.assertIsNone(result["condition_code"])
        self.assertFalse(result["mock"])
        self.assertTrue(result["limitations"])
        self.assertEqual(result["source"], weather_service.SOURCE_NAME)

    def test_a_broken_payload_reports_no_condition(self):
        with patch.object(
            weather_service.requests, "get", return_value=_FakeResponse({"data": {}})
        ):
            result = weather_service.get_weather(area="east")
        self.assertEqual(result["condition"], "unavailable")
        self.assertIsNone(result["condition_code"])

    def test_a_read_forecast_reports_the_condition_and_its_window(self):
        with patch.object(
            weather_service.requests, "get", return_value=_FakeResponse({"data": {"records": [FORECAST_RECORD]}})
        ):
            result = weather_service.get_weather(area="east")
        self.assertEqual(result["condition"], "Light rain")
        self.assertEqual(result["region"], "east")
        self.assertEqual(result["location"], "Bedok")
        self.assertEqual(result["as_of"], "2026-10-20T08:00:00+08:00")
        self.assertTrue(result["retrieved_at"])
        self.assertFalse(result["mock"])
        self.assertEqual(result["source_key"], "nea_mss_weather")

    def test_the_island_forecast_is_used_when_no_region_is_given(self):
        with patch.object(
            weather_service.requests, "get", return_value=_FakeResponse({"data": {"records": [FORECAST_RECORD]}})
        ):
            result = weather_service.get_weather()
        self.assertEqual(result["condition"], "Cloudy")
        self.assertIsNone(result["region"])
        self.assertEqual(result["island_forecast"], "Cloudy")

    def test_no_code_path_returns_a_condition_without_a_response(self):
        """A condition may only exist when a record was read."""
        for payload in (None, {}, {"data": {"records": []}}):
            with patch.object(
                weather_service.requests, "get", return_value=_FakeResponse(payload)
            ):
                result = weather_service.get_weather(area="east")
            self.assertEqual(result["condition"], "unavailable")

    def test_the_cache_keeps_the_same_forecast_for_a_second_call(self):
        response = _FakeResponse({"data": {"records": [FORECAST_RECORD]}})
        with patch.object(weather_service.requests, "get", return_value=response) as request:
            first = weather_service.get_weather(area="east")
            second = weather_service.get_weather(area="west")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(first["as_of"], second["as_of"])
        self.assertEqual(first["condition"], "Light rain")
        self.assertEqual(second["condition"], "Sunny")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
