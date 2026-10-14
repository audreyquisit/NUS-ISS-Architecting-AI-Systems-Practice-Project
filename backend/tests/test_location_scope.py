import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agents.location_agent import (
    LocationIntent,
    LocationResolutionError,
    MissingOriginError,
    _resolve_origin,
)
from models.location_models import LocationQuery
from pydantic import ValidationError
from services.location_service import (
    LocationServiceError,
    _base_url,
    geocode_place,
    is_singapore_location,
)


def _intent():
    return LocationIntent(
        request_intent="find_nearby_centres",
        origin_place=None,
        proximity_preference="convenient",
        interpretation="Find nearby centres.",
    )


class LocationScopeTests(unittest.TestCase):
    evaluations = []
    report_path = None

    @classmethod
    def setUpClass(cls):
        cls.evaluations = []
        report_dir = Path(__file__).resolve().parents[1] / "test-results"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cls.report_path = report_dir / f"location-scope-evaluation-{timestamp}.json"

    @classmethod
    def tearDownClass(cls):
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "suite": "location scope tests",
            "live_model_or_api_calls": False,
            "cases": cls.evaluations,
            "summary": {
                "total": len(cls.evaluations),
                "passed": sum(item["result"] == "PASS" for item in cls.evaluations),
                "failed": sum(item["result"] == "FAIL" for item in cls.evaluations),
            },
        }
        cls.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nLocation scope test report: {cls.report_path}")

    def assertEvaluation(self, inputs, expected, actual):
        passed = actual == expected
        record = {
            "test_case": self.id().rsplit(".", 1)[-1],
            "inputs": inputs,
            "expected": expected,
            "actual": actual,
            "result": "PASS" if passed else "FAIL",
        }
        self.__class__.evaluations.append(record)
        print("LOCATION_SCOPE_CASE " + json.dumps(record, ensure_ascii=False, default=str))
        self.assertEqual(actual, expected)

    @patch("agents.location_agent.reverse_geocode_location", return_value=None)
    @patch("agents.location_agent.is_singapore_location", return_value=True)
    def test_arbitrary_singapore_coordinates_are_accepted(
        self, _is_singapore, _reverse_geocode
    ):
        query = LocationQuery(
            request="Recommend food near my current location",
            user_location={"lat": 1.3521, "lng": 103.8198},
            user_location_source="browser_current_location",
        )

        origin, source = _resolve_origin(query, _intent())

        self.assertEvaluation(
            {"location_query": query.model_dump(), "reverse_geocode_result": None},
            {
                "origin": {
                    "lat": 1.3521,
                    "lng": 103.8198,
                    "label": "your current location",
                    "source": "browser_current_location",
                },
                "source": "browser_current_location",
            },
            {"origin": origin, "source": source},
        )

    @patch("agents.location_agent.is_singapore_location", return_value=False)
    def test_coordinates_outside_singapore_are_rejected(self, _is_singapore):
        query = LocationQuery(
            request="Recommend food near my current location",
            # Inside the broad pre-validation box, but mocked reverse-geocode
            # evidence says this point is not in Singapore.
            user_location={"lat": 1.55, "lng": 104.2},
            user_location_source="browser_current_location",
        )

        try:
            _resolve_origin(query, _intent())
            actual = {"raised": None}
        except LocationResolutionError as error:
            actual = {"raised": type(error).__name__, "message": str(error)}

        self.assertEvaluation(
            {"location_query": query.model_dump(), "singapore_validation": False},
            {
                "raised": "LocationResolutionError",
                "message": (
                    "That starting point appears to be outside Singapore. "
                    "Please provide a location within Singapore."
                ),
            },
            actual,
        )

    def test_coordinates_outside_service_area_fail_before_reverse_geocoding(self):
        try:
            LocationQuery(
                request="Find food near me",
                user_location={"lat": 3.139, "lng": 101.6869},
            )
            actual = {"rejected": False}
        except ValidationError as error:
            actual = {
                "rejected": True,
                "fields": sorted(".".join(map(str, item["loc"])) for item in error.errors()),
            }
        self.assertEvaluation(
            {"coordinates": {"lat": 3.139, "lng": 101.6869}},
            {"rejected": True, "fields": ["user_location.lat", "user_location.lng"]},
            actual,
        )

    def test_place_query_over_limit_is_rejected_before_onemap_call(self):
        long_place = "T" * 201
        with patch("services.location_service._get") as get:
            try:
                geocode_place(long_place)
                actual = {"rejected": False, "onemap_called": get.called}
            except LocationServiceError as error:
                actual = {
                    "rejected": True,
                    "error": str(error),
                    "onemap_called": get.called,
                }
        self.assertEvaluation(
            {"place_query_chars": len(long_place), "maximum_chars": 200},
            {
                "rejected": True,
                "error": "Place search must be 200 characters or fewer.",
                "onemap_called": False,
            },
            actual,
        )

    def test_onemap_host_configuration_rejects_untrusted_host(self):
        with patch.dict("os.environ", {"ONEMAP_BASE_URL": "https://attacker.example"}):
            try:
                actual = {"base_url": _base_url(), "rejected": False}
            except LocationServiceError as error:
                actual = {"error": str(error), "rejected": True}
        self.assertEvaluation(
            {"configured_base_url": "https://attacker.example"},
            {
                "rejected": True,
                "error": "ONEMAP_BASE_URL must use https://www.onemap.gov.sg.",
            },
            actual,
        )

    @patch("agents.location_agent.reverse_geocode_location", return_value="Jurong Point")
    @patch("agents.location_agent.geocode_place", return_value={"lat": 1.339, "lng": 103.706})
    def test_explicit_new_origin_replaces_carried_origin(self, geocode, _reverse):
        query = LocationQuery(
            request="Food near Jurong Point",
            location_text="Jurong Point",
            conversation_context={"state": {"location_text": "Clementi"}},
        )
        origin, source = _resolve_origin(query, _intent())
        actual = {
            "geocode_query": geocode.call_args.args[0],
            "origin_label": origin["label"],
            "source": source,
        }
        self.assertEvaluation(
            {
                "current_message": query.request,
                "manual_origin": query.location_text,
                "prior_origin": "Clementi",
            },
            {
                "geocode_query": "Jurong Point",
                "origin_label": "Jurong Point",
                "source": "free_text_place",
            },
            actual,
        )

    @patch("agents.location_agent.reverse_geocode_location", return_value="Current Singapore location")
    @patch("agents.location_agent.is_singapore_location", return_value=True)
    @patch("agents.location_agent.geocode_place")
    def test_near_me_uses_current_coordinates_instead_of_stale_place(
        self, geocode, _is_singapore, _reverse
    ):
        query = LocationQuery(
            request="Find food near me",
            user_location={"lat": 1.3521, "lng": 103.8198},
            user_location_source="browser_current_location",
            conversation_context={
                "state": {
                    "location_text": "Clementi",
                    "origin_source": "free_text_place",
                }
            },
        )
        origin, source = _resolve_origin(query, _intent())
        actual = {
            "source": source,
            "origin_coordinates": {"lat": origin["lat"], "lng": origin["lng"]},
            "geocode_called": geocode.called,
        }
        self.assertEvaluation(
            {"current_message": query.request, "prior_origin": "Clementi"},
            {
                "source": "browser_current_location",
                "origin_coordinates": {"lat": 1.3521, "lng": 103.8198},
                "geocode_called": False,
            },
            actual,
        )

    @patch("agents.location_agent.geocode_place")
    def test_broad_region_is_not_used_as_a_route_origin(self, geocode):
        query = LocationQuery(
            request="Find cheap food in the East",
            parsed_request={
                "search_scope_mode": "region",
                "requested_area": "east",
            },
        )
        try:
            _resolve_origin(query, _intent())
            actual = {"requested_origin": None, "geocode_called": geocode.called}
        except MissingOriginError as error:
            actual = {"requested_origin": "required", "geocode_called": geocode.called}
        self.assertEvaluation(
            {"requested_area": "east", "explicit_origin": None},
            {"requested_origin": "required", "geocode_called": False},
            actual,
        )

    @patch("agents.location_agent.reverse_geocode_location", return_value="Clementi MRT Station")
    @patch("agents.location_agent.geocode_place", return_value={"lat": 1.315, "lng": 103.765})
    def test_misspelled_station_is_corrected_to_catalogue_name(self, geocode, _reverse):
        query = LocationQuery(
            request="Food near Cementi",
            location_text="Cementi",
        )
        origin, _source = _resolve_origin(query, _intent())
        actual = {
            "geocode_query": geocode.call_args.args[0],
            "display_label": origin["label"],
        }
        self.assertEvaluation(
            {"typed_place": "Cementi", "station_catalogue_match": "Clementi"},
            {
                "geocode_query": "Clementi MRT Station",
                "display_label": "Clementi MRT Station",
            },
            actual,
        )

    @patch("services.location_service._get")
    def test_reverse_geocode_evidence_controls_singapore_validation(self, get):
        point = {"lat": 1.3521, "lng": 103.8198}
        get.return_value = {"GeocodeInfo": [{"POSTALCODE": "123456"}]}
        is_singapore = is_singapore_location(point)
        endpoint = get.call_args_list[0].args[0]
        buffer_m = get.call_args_list[0].kwargs["params"]["buffer"]

        get.return_value = {"GeocodeInfo": []}
        outside_singapore = is_singapore_location(
            {"lat": 3.139, "lng": 101.6869}
        )
        self.assertEvaluation(
            {
                "points": [point, {"lat": 3.139, "lng": 101.6869}],
                "mocked_geocode_info": [[{"POSTALCODE": "123456"}], []],
            },
            {
                "endpoint": "https://www.onemap.gov.sg/api/public/revgeocode",
                "buffer_m": 250,
                "inside_point_is_singapore": True,
                "outside_point_is_singapore": False,
            },
            {
                "endpoint": endpoint,
                "buffer_m": buffer_m,
                "inside_point_is_singapore": is_singapore,
                "outside_point_is_singapore": outside_singapore,
            },
        )


if __name__ == "__main__":
    unittest.main()
