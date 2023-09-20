from .log_handler import LogHandler


_log_handler = LogHandler()


def get_logger():
    return _log_handler.get_logger()
