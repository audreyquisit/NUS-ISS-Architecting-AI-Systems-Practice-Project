from typing import List, Dict, Any


# Verification helpers

def verify_location_result(
    result: Dict[str, Any],
    user_request: Dict[str, Any]
) -> Dict[str, Any]:

    verified_candidates = []

    max_time = user_request.get(
        "available_time_minutes"
    )

    for candidate in result.get("candidates", []):

        distance = candidate.get(
            "distance_minutes"
        )

        if distance is None:
            continue

        # Reject candidates that take longer than the user's available time.

        if max_time is not None and distance > max_time:
            continue

        verified_candidates.append(candidate)

    return {
        "agent": "location",
        "verified": True,
        "candidates": verified_candidates
    }


def verify_budget_result(
    result: Dict[str, Any],
    user_request: Dict[str, Any]
) -> Dict[str, Any]:

    verified_candidates = []

    budget = user_request.get(
        "budget"
    )

    for candidate in result.get("candidates", []):

        price = candidate.get("price")

        if price is None:
            continue

        # Do not allow the recommendation agent to receive candidates that exceed the budget.

        if price <= budget:
            verified_candidates.append(candidate)

    return {
        "agent": "budget",
        "verified": True,
        "candidates": verified_candidates
    }


def verify_dietary_result(
    result: Dict[str, Any],
    user_request: Dict[str, Any]
) -> Dict[str, Any]:

    dietary_preference = user_request.get(
        "dietary_preference"
    )

    # If no dietary requirement was provided, there is nothing to verify.

    if not dietary_preference:
        return {
            "agent": "dietary",
            "verified": True,
            "candidates": result.get("candidates", [])
        }

    verified_candidates = []

    for candidate in result.get("candidates", []):

        dietary_suitable = candidate.get(
            "dietary_suitable"
        )

        if dietary_suitable is True:
            verified_candidates.append(candidate)

    return {
        "agent": "dietary",
        "verified": True,
        "candidates": verified_candidates
    }


def verify_queue_result(
    result: Dict[str, Any],
    user_request: Dict[str, Any]
) -> Dict[str, Any]:

    verified_candidates = []

    available_time = user_request.get(
        "available_time_minutes"
    )

    for candidate in result.get("candidates", []):

        queue_minutes = candidate.get(
            "queue_minutes"
        )

        if queue_minutes is None:
            continue

        # A queue longer than the available time should not be recommended.

        if queue_minutes <= available_time:
            verified_candidates.append(candidate)

    return {
        "agent": "queue",
        "verified": True,
        "candidates": verified_candidates
    }


def verify_weather_result(
    result: Dict[str, Any],
    user_request: Dict[str, Any]
) -> Dict[str, Any]:

    weather = result.get(
        "weather"
    )

    # Weather information should exist before allowing the recommendation agent to use it.

    if not weather:
        return {
            "agent": "weather",
            "verified": False,
            "reason": "No weather information available."
        }

    return {
        "agent": "weather",
        "verified": True,
        "weather": weather
    }


# Main verifier

def verify_results(
    agent_results: List[Dict[str, Any]],
    user_request: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Verify results returned by specialised agents.

    Invalid or unsupported candidates are removed before
    the results are passed to the recommendation agent.
    """

    verified_results = []

    for result in agent_results:

        agent = result.get("agent")

        if agent == "location":

            verified = verify_location_result(
                result,
                user_request
            )

        elif agent == "dietary":

            verified = verify_dietary_result(
                result,
                user_request
            )

        elif agent == "budget":

            verified = verify_budget_result(
                result,
                user_request
            )

        elif agent == "queue":

            verified = verify_queue_result(
                result,
                user_request
            )

        elif agent == "weather":

            verified = verify_weather_result(
                result,
                user_request
            )

        else:

            # Unknown agent results should notautomatically be trusted.

            verified = {
                "agent": agent,
                "verified": False,
                "reason": "Unknown agent."
            }

        verified_results.append(
            verified
        )

    return verified_results
