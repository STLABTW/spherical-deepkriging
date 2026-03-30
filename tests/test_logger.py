import logging

from spherical_deepkriging.logger import setup_logger


def test_setup_logger_creates_handler_and_supports_set_level():
    logger_name = "spherical_deepkriging.tests.logger.unique"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    configured = setup_logger(logger_name, level=logging.WARNING)

    assert configured.name == logger_name
    assert configured.level == logging.WARNING
    assert len(configured.handlers) == 1

    configured.set_level(logging.ERROR)

    assert configured.level == logging.ERROR
    assert configured.handlers[0].level == logging.ERROR


def test_setup_logger_reuses_existing_logger_handlers():
    logger_name = "spherical_deepkriging.tests.logger.reuse"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    reused = setup_logger(logger_name)

    assert reused is logger
    assert len(reused.handlers) == 1
