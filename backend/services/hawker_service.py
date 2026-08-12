def find_nearby_stalls(location: str):

    return [
        {
            "stall_name": "Example Vegetarian Stall",
            "hawker_centre": "Example Hawker Centre",
            "distance_minutes": 5,
            "price": 6.50
        },
        {
            "stall_name": "Example Noodle Stall",
            "hawker_centre": "Example Hawker Centre",
            "distance_minutes": 8,
            "price": 7.00
        }
    ]


def find_dietary_stalls(
    dietary_preference: str
):

    return [
        {
            "stall_name": "Example Vegetarian Stall",
            "dietary_suitable": True
        }
    ]


def find_stalls_within_budget(
    budget: float
):

    return [
        {
            "stall_name": "Example Vegetarian Stall",
            "price": 6.50
        }
    ]