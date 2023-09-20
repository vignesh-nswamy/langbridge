import logging


class LogHandler:
    def __init__(self):
        self.logger = logging.getLogger("openai-processor")
        self.logger.setLevel(logging.INFO)
        try:
            import rich
        except ImportError:
            formatter = logging.Formatter(f'["openai-processor"] - %(asctime)s - %(levelname)s - %(message)s')

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)

            self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
