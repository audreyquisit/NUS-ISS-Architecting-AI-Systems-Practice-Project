from services.hawker_service import find_dietary_stalls


def run(task: str):

    print("Dietary Agent")

    stalls = find_dietary_stalls(
        dietary_preference="vegetarian"
    )

    return {
        "agent": "dietary",
        "task": task,
        "candidates": stalls
    }
