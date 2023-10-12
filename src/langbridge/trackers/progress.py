from dataclasses import dataclass


@dataclass
class ProgressTracker:
    """
    Stores metadata about the progress of API calls.
    """
    # Tasks
    num_tasks_initiated: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    # Errors
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    # Time
    time_last_rate_limit_error: int = 0
