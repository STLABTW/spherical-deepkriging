import logging


def setup_logger(
    name: str = "spherical_deepkriging.logger", level: int = logging.INFO
) -> logging.Logger:
    if logging.getLogger(name).handlers:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def set_level(new_level):
        logger.setLevel(new_level)
        console_handler.setLevel(new_level)
        logger.info(f"Logging level changed to {logging.getLevelName(new_level)}")

    logger.set_level = set_level

    return logger
