from schemas import StallRecommendation, UserPreferences


def generate_recommendation(
    preferences: UserPreferences,
) -> StallRecommendation:
    """Generate a stub recommendation.

    Replace with actual AI agent orchestration logic.
    """
    return StallRecommendation(
        stall_name="Example Noodle Stall",
        hawker_centre="Maxwell Food Centre",
        cuisine="Chinese",
        price_range="Low",
        score=0.94,
        explanation=(
            "This recommendation is based on your location, budget, and "
            "dietary preferences. Replace this stub with real recommendation "
            "logic when integrating agents."
        ),
    )
