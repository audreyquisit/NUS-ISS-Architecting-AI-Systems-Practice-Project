from services.live_queue_feed import get_live_queue_feed


def run(task: str):

    print("Queue Agent")

    queue_data = get_live_queue_feed(task=task)

    return {
        "agent": "queue",
        "task": task,
        "candidates": queue_data
    }
