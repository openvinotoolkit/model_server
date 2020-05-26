import logging

import config

LOGGER_LEVEL = "INFO"


def get_logger(name):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.setLevel(config.log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    return logger
