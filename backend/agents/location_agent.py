from services.hawker_service import find_nearby_stalls


def run(task: str):

    print("Location Agent")

    stalls = find_nearby_stalls(
        location="Raffles Place"
    )

    return {
        "agent": "location",
        "task": task,
        "candidates": stalls
    }