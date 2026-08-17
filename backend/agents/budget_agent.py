from services.hawker_service import find_stalls_within_budget


def run(task: str):

    print("Budget Agent")

    stalls = find_stalls_within_budget(
        budget=8
    )

    return {
        "agent": "budget",
        "task": task,
        "candidates": stalls
    }
