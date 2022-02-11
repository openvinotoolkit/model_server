from cmath import log
import logging

available_log_levels = ['INFO', 'DEBUG', 'ERROR']

def get_logger(name):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
    logger.setLevel(LoggerConfig.log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    return logger

class LoggerConfig:
    log_level = "INFO"

    @classmethod
    def set_log_level(cls, log_level: str):
        log_level = log_level.upper()
        if log_level in available_log_levels:
            cls.log_level = log_level
        else:
            cls.log_level = "INFO"
