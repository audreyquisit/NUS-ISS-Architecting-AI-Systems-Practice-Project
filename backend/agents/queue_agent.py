from services.queue_service import get_queue_estimates


def run(task: str):

    print("Queue Agent")

    queue_data = get_queue_estimates()

    return {
        "agent": "queue",
        "task": task,
        "candidates": queue_data
    }
