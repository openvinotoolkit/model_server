import logging

available_lvl_of_logger = ['INFO', 'DEBUG', 'ERROR']

def get_logger_lvl():
    requested_lvl = "INFO"
    requested_lvl = requested_lvl.upper()
    global LOGGER_LVL
    if requested_lvl in available_lvl_of_logger:
        return requested_lvl
    return 'INFO'


LOGGER_LVL = get_logger_lvl()


def get_logger(name):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
    logger.setLevel(LOGGER_LVL)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    return logger